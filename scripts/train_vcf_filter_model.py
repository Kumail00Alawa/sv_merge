from modules.shallow_linear import ShallowLinear
from modules.data_loader import VcfDatasetBNDs
from collections import defaultdict
from torch import multiprocessing

# from modules.misc import plot_roc_curve, write_recalibrated_vcf, plot_confusion

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from matplotlib import pyplot

import numpy as np

from datetime import datetime
from collections import OrderedDict
from copy import deepcopy
import random
import numpy
import vcfpy
import torch
import math
import sys
import os

# set matplotlib to use the Agg backend
pyplot.switch_backend('Agg')


print("USING pytorch VERSION: ", torch.__version__)


def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs, max_batches_per_epoch):
    train_losses = list()
    test_losses = list()
    test_aucs = list()
    models = list()

    model.eval()
    y_predict, y_true, test_loss = test(model=model, loader=test_loader, loss_fn=loss_fn)
    test_losses.append(test_loss)
    test_aucs.append(roc_auc_score(y_true=y_true, y_score=y_predict))
    models.append(model.state_dict())
    model.train()

    n_total_positive = 0
    n_total_negative = 0

    prev_model = None

    batch_index = 0
    for e in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            if i >= max_batches_per_epoch:
                break

            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn)

            n_positive = sum(y.data.numpy())
            n_negative = len(y) - n_positive

            print(i,n_positive,n_negative)

            n_total_positive += n_positive
            n_total_negative += n_negative

            train_losses.append(loss)

            batch_index += 1

        if e % 1 == 0:
            model.eval()
            y_predict, y_true, test_loss = test(model=model, loader=test_loader, loss_fn=torch.nn.BCELoss(), use_sigmoid=True)

            auc = roc_auc_score(y_true=y_true, y_score=y_predict)

            test_losses.append(test_loss)
            test_aucs.append(auc)
            models.append(deepcopy(model.state_dict()))
            model.train()

            if len(test_losses) > 1:
                test_loss_worsened = test_losses[-1] > test_losses[-2]
                test_auc_worsened = test_aucs[-1] < test_aucs[-2]

                print(f"Epoch:{e}\tbatches:{batch_index}\tlr_rate:{scheduler.get_last_lr()}\tn_positive:{n_total_positive}\tn_negative:{n_total_negative}\tprev_loss:{test_losses[-2]:.3f}\tloss:{test_losses[-1]:.3f}\tworsened:{test_loss_worsened}\tprev_auc:{test_aucs[-2]:.3f}\tauc:{test_aucs[-1]:.3f}\tworsened:{test_auc_worsened}")

    # Get index of greatest AUC
    max_auc_index = test_aucs.index(max(test_aucs))

    # Get the model with the greatest AUC
    model.load_state_dict(models[max_auc_index])

    return train_losses, test_losses, test_aucs


def train_batch(model, x, y, optimizer, scheduler, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # convert 1-hot vectors back into indices
    # max_values, target_index = y.max(dim=1)
    # target_index = target_index.type(torch.LongTensor)

    # Compute loss.
    loss = loss_fn(y_predict.squeeze(), y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    return loss.data.item()


def test(model, loader, loss_fn, use_sigmoid=False):
    y_vectors = list()
    y_predict_vectors = list()
    losses = list()

    batch_index = 0
    for x, y in loader:
        y_predict = model.forward(x, use_sigmoid=use_sigmoid)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

        loss = loss_fn(y_predict.squeeze(), y)
        losses.append(loss.data.numpy())

    y_predict_vector = np.concatenate(y_predict_vectors)
    y_vector = np.concatenate(y_vectors)

    loss = numpy.sum(losses)

    return y_predict_vector, y_vector, loss


def plot_loss(train_losses, test_losses, test_aucs, label, output_dir=None):
    fig,ax = pyplot.subplots(nrows=3)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Test Loss")
    ax[2].set_ylabel("Test AUC")

    x_train = list(range(len(train_losses)))
    x_test = list(range(len(test_losses)))

    ax[0].plot(x_train, train_losses, label="train")
    ax[1].plot(x_test, test_losses, label="test")
    ax[2].plot(x_test, test_aucs, label="test")

    fig.suptitle(label)

    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, label + "_loss.png"), dpi=200)


def run(dataset_train, dataset_test, output_dir, downsample=False):
    # Define the hyperparameters
    learning_rate_max = 1
    learning_rate_base = 1e-5
    weight_decay = 4e-6
    dropout_rate = 0.01

    goal_batch_size = 1024 #4096 (previous batch size)
    n_epochs = 16

    config_path = os.path.join(output_dir, "config.txt")

    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = min(goal_batch_size, len(dataset_train))

    max_batches_per_epoch = sys.maxsize

    if downsample:
        minority_class_size = int(min(sum(dataset_train.y_data == 0), sum(dataset_train.y_data == 1)))
        batch_size_train = int(min(goal_batch_size, minority_class_size*2))

        # Force the data loader to stop iterating once the minority class is expected to be fully iterated.
        # This is mostly just an inconvenience with how PyTorch structures its DataLoader with WeightedRandomSampler
        # TODO: for future use remove the *2 factor which assumes two (binary) labels
        max_batches_per_epoch = max(1, ((minority_class_size*2)/batch_size_train) - 1)

    print("Using batch size: ", batch_size_train, "minority_class_size: ", minority_class_size, "max_batches_per_epoch: ", max_batches_per_epoch)

    # write the hyperparameters to a file
    with open(config_path, "w") as f:
        f.write("learning_rate_base: {}\n".format(learning_rate_base))
        f.write("learning_rate_max: {}\n".format(learning_rate_max))
        f.write("weight_decay: {}\n".format(weight_decay))
        f.write("n_epochs: {}\n".format(n_epochs))
        f.write("goal_batch_size: {}\n".format(goal_batch_size))
        f.write("downsample: {}\n".format(downsample))
        f.write("batch_size_train: {}\n".format(batch_size_train))
        f.write("dropout_rate: {}\n".format(dropout_rate))
        if dataset_train.filter_fn is not None:
            f.write("filter_fn: {}\n".format(str(dataset_train.filter_fn.__name__)))
        else:
            f.write("filter_fn: None\n")
        f.write("data_conditions: {}\n".format(dataset_train.data_conditions))

    data_loader_train = None

    if downsample:
        weight_tensor = compute_downsampling_weight_tensor(dataset_train)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_tensor, len(weight_tensor), replacement=True)
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, sampler=sampler)
    else:
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)

    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    input_size = len(dataset_train[0][0])
    print("using input size: ", input_size)
    shallow_model = ShallowLinear(input_size=input_size, dropout_rate=dropout_rate)

    # Initialize the optimizer with above parameters
    optimizer = optim.SGD(shallow_model.parameters(), lr=learning_rate_base, weight_decay=weight_decay)

    # Initialize CLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, steps_per_epoch=len([x for x in data_loader_train]), epochs=n_epochs)

    # Define the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Train and get the resulting loss per iteration
    train_losses, test_losses, test_aucs = train(model=shallow_model, train_loader=data_loader_train, test_loader=data_loader_test, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, epochs=n_epochs, max_batches_per_epoch=max_batches_per_epoch)

    # Test and get the resulting predicted y values
    shallow_model.eval() # switch to eval mode to disable dropout

    return train_losses, test_losses, test_aucs, shallow_model


def compute_downsampling_weight_tensor(dataset):
    target_epoch_size = 50_000

    y_data = dataset.y_data

    n_total = len(y_data)
    n_true = sum(y_data)
    n_false = n_total - sum(y_data)
    weight_tensor = torch.clone(y_data).detach()
    weight = None

    # compute a weight such that the majority class will be downsampled by that weight to match the minority weight,
    # and then construct a tensor of weights such that each element of the minority class has weight 1 and each element
    # of the majority class has weight equal to the downsample weight
    if n_true < n_false:
        weight = n_true / n_false
        weight_tensor[y_data == 0] = weight
        weight_tensor[y_data == 1] = 1
    else:
        weight = n_false / n_true
        weight_tensor[y_data == 1] = weight
        weight_tensor[y_data == 0] = 1

    expected_epoch_size = float(torch.sum(weight_tensor))

    print("n_true: %d\tn_false: %d\texpected_epoch_size: %d\tmajority class weight: %d" % (n_true, n_false, expected_epoch_size, weight))

    return weight_tensor


'''
Downsamples the test data to have an equal number of true and false labels. Is a hard downsample, used in testing,
whereas the training data is downsampled using weighted random sampling.
'''
def downsample_test_data(y_true, y_predict, seed=37):
    random.seed(seed)

    n_true = sum(y_true)
    n_false = len(y_true) - n_true

    true_weight = min(1.0, float(n_false) / float(n_true))
    false_weight = min(1.0, float(n_true) / float(n_false))

    mask = numpy.zeros(len(y_true))

    for i in range(len(y_true)):
        if y_true[i] == 1 and random.random() <= true_weight:
            mask[i] = 1
        if y_true[i] == 0 and random.random() <= false_weight:
            mask[i] = 1

    y_true = y_true[mask == 1]
    y_predict = y_predict[mask == 1]

    return y_true, y_predict


def plot_roc_curve(y_true, y_predict, label, axes, color=None, alpha=1.0, style='-', use_text=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)

    if color is not None:
        if label is not None:
            axes.plot(fpr, tpr, label=label, color=color, linestyle=style, alpha=alpha)
        else:
            axes.plot(fpr, tpr, color=color, linestyle=style, alpha=alpha)

    else:
        if label is not None:
            axes.plot(fpr, tpr, label=label, linestyle=style, alpha=alpha)
        else:
            axes.plot(fpr, tpr, linestyle=style, alpha=alpha)

        color = axes.get_lines()[-1].get_color()

    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")

    axes.grid(which='major', color='black', linestyle='-', linewidth=0.5)
    axes.grid(which='minor', color='black', linestyle=':', linewidth=0.5)

    if use_text:
        l = len(thresholds)
        n = 10
        interval = max(1,(l//n))

        for i,[x,y,t] in enumerate(zip(fpr, tpr, thresholds)):
            # only label 1/10 of the points
            if i % interval == 0:
                axes.text(x, y, round(t, 2), fontsize='small')
                # add marker
                axes.plot(x, y, marker='o', color=color, markeredgewidth=0)

    return axes, fpr, tpr, thresholds


def write_filtered_vcf(y_predict, records, threshold, input_vcf_path, output_vcf_path):
    reader = vcfpy.Reader.from_path(input_vcf_path)

    writer = vcfpy.Writer.from_path(output_vcf_path, reader.header)

    for record, score in zip(records, y_predict):
        if score > threshold:
            writer.write_record(record)


def min50bp(record):
    if type(record.INFO["SVLEN"]) == list:
        return record.INFO["SVLEN"][0] >= 50
    else:
        return record.INFO["SVLEN"] >= 50


def thread_fn(truth_info_name, annotation_name, train_vcfs, train_contigs, test_vcfs, test_contigs, eval_vcfs, eval_contigs, filter_fn, output_dir, data_conditions):
    print("Thread started with ", truth_info_name, annotation_name)

    label = truth_info_name + " truth labels and " + annotation_name + " features"

    dataset_train = VcfDatasetBNDs(vcf_paths=train_vcfs, truth_info_name=truth_info_name, annotation_name=annotation_name, contigs=train_contigs, filter_fn=filter_fn, data_conditions=data_conditions, evaluate_complex_bnds=False)
    dataset_test = VcfDatasetBNDs(vcf_paths=test_vcfs, truth_info_name=truth_info_name, annotation_name=annotation_name, contigs=test_contigs, filter_fn=filter_fn, data_conditions=data_conditions, evaluate_complex_bnds=False)
    dataset_eval = VcfDatasetBNDs(vcf_paths=eval_vcfs, truth_info_name=truth_info_name, annotation_name=annotation_name, contigs=eval_contigs, filter_fn=filter_fn, data_conditions=data_conditions, evaluate_complex_bnds=data_conditions['evaluate_complex_bnds'])

    n_train = len(dataset_train)
    n_test = len(dataset_test)

    # Print information on the train and test datasets
    print('Train datapoints: ', n_train)
    print('Test datapoints: ', n_test)
    print('Input shape: ', dataset_train[0][0].shape)
    print('Output shape: ', dataset_train[0][1].shape)

    train_losses, test_losses, test_aucs, model = run(dataset_train=dataset_train, dataset_test=dataset_test, downsample=True, output_dir=output_dir)

    # Format the date and time
    model_output_path = os.path.join(output_dir, label.replace(" ", "_") + ".pt")
    torch.save(model.state_dict(), model_output_path)

    # calculate and plot loss
    print("Final loss:", sum(train_losses[-100:])/100)
    plot_loss(train_losses=train_losses, test_losses=test_losses, test_aucs=test_aucs, label=label, output_dir=output_dir)

    data_loader_eval = DataLoader(dataset=dataset_eval, batch_size=len(dataset_eval), shuffle=False)

    x = dataset_eval.x_data.numpy()

    # Finally evaluate on a 3rd dataset which is not used to select training termination conditions
    # Need to use_sigmoid this time to get meaningful 0-1 values
    y_predict, y_true, mean_loss = test(model=model, loader=data_loader_eval, loss_fn=torch.nn.BCELoss(), use_sigmoid=True)

    return dataset_eval.records, x, dataset_eval.feature_indexes, y_true, y_predict, truth_info_name, annotation_name


def plot_tandem_stratified_roc_curves(axes, records, y_true, y_predict, truth_info_name, annotation_name):
    tandem = [[],[]]
    non_tandem = [[],[]]

    for r,record in enumerate(records):
        is_tandem = record.INFO["tr_coverage"] > 0.9

        if is_tandem:
            tandem[0].append(y_true[r])
            tandem[1].append(y_predict[r])
        else:
            non_tandem[0].append(y_true[r])
            non_tandem[1].append(y_predict[r])

    auc_tandem = round(roc_auc_score(y_true=tandem[0], y_score=tandem[1]), 6)
    auc_non_tandem = round(roc_auc_score(y_true=non_tandem[0], y_score=non_tandem[1]), 6)

    label = truth_info_name + " truth labels and " + annotation_name + " features TANDEM, " + "AUC = " + str(auc_tandem)
    axes, fpr, tpr, thresholds = plot_roc_curve(y_true=tandem[0], y_predict=tandem[1], label=label, axes=axes, style=':')

    color = axes.get_lines()[-1].get_color()

    label = truth_info_name + " truth labels and " + annotation_name + " features NON-TANDEM, " + "AUC = " + str(auc_non_tandem)
    axes, fpr, tpr, thresholds = plot_roc_curve(y_true=non_tandem[0], y_predict=non_tandem[1], label=label, color=color, axes=axes)

    return axes


def get_length(record: vcfpy.Record):
    if "SVLEN" in record.INFO:
        if type(record.INFO["SVLEN"]) == list:
            return abs(record.INFO["SVLEN"][0])
        else:
            return abs(record.INFO["SVLEN"])
    else:
        return abs(len(record.ALT[0]) - len(record.REF))


def plot_length_stratified_roc_curves(axes, records, y_true, y_predict, truth_info_name, annotation_name, tandem_only=False):
    length_stratified_results = defaultdict(lambda: [[],[]])
    colormap = pyplot.colormaps['viridis']

    for r,record in enumerate(records):
        length_bin = int(math.log2(get_length(record) + 1))

        if tandem_only and record.INFO["tr_coverage"] <= 0.9:
            continue

        length_stratified_results[length_bin][0].append(y_true[r])
        length_stratified_results[length_bin][1].append(y_predict[r])

    for i,(l,result) in enumerate(sorted(length_stratified_results.items(), key=lambda x: x[0])):
        bin_start = 2**(l)
        bin_end = 2**(l+1)
        label = truth_info_name + " truth labels and " + annotation_name + " features " + str(bin_start) + " to " + str(bin_end) + " bp"

        f = float(i)/float(len(length_stratified_results))
        color = colormap(f)
        axes, fpr, tpr, thresholds = plot_roc_curve(y_true=result[0], y_predict=result[1], label=label, axes=axes, color=color, alpha = 0.7)

    return axes


def write_vcf_config(output_dir, train_vcfs, test_vcfs, eval_vcfs, train_contigs, test_contigs, eval_contigs):
    with open(os.path.join(output_dir, "train_vcfs.txt"), "w") as f:
        for path in train_vcfs:
            f.write(path + "\n")

    with open(os.path.join(output_dir, "test_vcfs.txt"), "w") as f:
        for path in test_vcfs:
            f.write(path + "\n")

    with open(os.path.join(output_dir, "train_contigs.txt"), "w") as f:
        for contig in train_contigs:
            f.write(contig + "\n")

    with open(os.path.join(output_dir, "test_contigs.txt"), "w") as f:
        for contig in test_contigs:
            f.write(contig + "\n")

    with open(os.path.join(output_dir, "eval_vcfs.txt"), "w") as f:
        for path in eval_vcfs:
            f.write(path + "\n")

    with open(os.path.join(output_dir, "eval_contigs.txt"), "w") as f:
        for contig in eval_contigs:
            f.write(contig + "\n")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_dir = os.path.join("../../output/sv_merge/", timestamp)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_vcfs = [
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG00621.vcf.gz", #00621
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG01928.vcf.gz", #01928
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG02572.vcf.gz", #02572
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG03098.vcf.gz", #03098
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG03492.vcf.gz", #03492
        # "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG002.vcf.gz", #002
        # "../data/hprc_08x_TEST_5_min_sv_length_2k/HG00438.vcf.gz", #00438
    ]

    test_vcfs = [
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG00673.vcf.gz", #00673
        # "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG00733.vcf.gz", #00733
        # "../data/hprc_08x_TEST_5_min_sv_length_2k/HG005.vcf.gz", #005
    ]

    eval_vcfs = [
        "../../data/sv_merge/hprc_08x_TEST_5_min_sv_length_2k/HG03516.vcf.gz", #03516
    ]

    train_contigs = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr21", "chr22", "chrX"}
    test_contigs = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr21", "chr22", "chrX"}
    eval_contigs = {"chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"}

    # Which truth info and annotation names to use (these names are hardcoded in the dataloader to define its behavior)
    truth_info_names = ["Hapestry"]
    # truth_info_names = ["Hapestry", "TruvariBench_TP"]
    annotation_names = ["Hapestry"]
    # annotation_names = ["Hapestry", "Sniffles"]

    # Whether to subset the VCFs to >= 50bp
    # filter_fn = min50bp
    filter_fn = None

    # Set data processing conditions
    data_conditions = {
    # Setting 'simple_bnds' to False will ignore the the condition of 'evaluate_bnds'
    'all_bnds' : True, # When set to true, the code will include all types of BNDs (simple and complex) for training, testing, and evaluation datasets. Otherwise, the code will only consider complex BNDs - Currently the code only considers the actual BNDs
    'evaluate_complex_bnds' : True, # When set to true, the code will only evaluate complex BNDs (this only affects evaluation) - Currently the code only considers the actual BNDs
    # Only set one of the following conditions as true at a time
    'concatenate_bnds' : False, # When set to true, the code will concatenate ID & MATEID data
    'max_bnds' : True, # When set to true, the code will obtain the maximum of ID & MATEID data
    'sum_bnds' : False, # When set to true, the code will add the data of ID & MATEID
    'average_bnds' : False, # When set to true, the code will obtain average the data of ID & MATEID
    'features' : False, # When set to true, the code will include all of the features of a given record. The example of the features: BND Connection such as: intra, inter, and single BND & BND Direction: +/+, +/-, -/+, -/- 
    }

    # Write a bunch of config files for record keeping
    write_vcf_config(output_dir, train_vcfs, test_vcfs, eval_vcfs, train_contigs, test_contigs, eval_contigs)

    # Initialize some things for plotting
    fig = pyplot.figure()
    fig.set_size_inches(10,8)
    axes = pyplot.axes()

    roc_data = dict()

    tandem_fig = pyplot.figure()
    tandem_fig.set_size_inches(10,8)
    tandem_axes = pyplot.axes()

    length_figs = dict()
    length_axes = dict()

    n_processes = len(truth_info_names) * len(annotation_names)

    args = list()

    # Set up args for multithreading (each combo of truth/label will get its own thread)
    for truth_info_name in truth_info_names:
        for annotation_name in annotation_names:
            args.append((truth_info_name, annotation_name, train_vcfs, train_contigs, test_vcfs, test_contigs, eval_vcfs, eval_contigs, filter_fn, output_dir, data_conditions))
            length_figs[truth_info_name + "_" + annotation_name] = pyplot.figure(figsize=(10,8))
            length_axes[truth_info_name + "_" + annotation_name] = pyplot.axes()

    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(thread_fn, args)

    # results = [thread_fn(truth_info_names[0], annotation_names[0], train_vcfs, train_contigs, test_vcfs, test_contigs, eval_vcfs, eval_contigs, filter_fn, output_dir)]

    for r,[records, x, feature_indexes, y_true, y_predict, truth_info_name, annotation_name] in enumerate(results):
        label = truth_info_name + " truth labels and " + annotation_name + " features"
        length_axis = length_axes[truth_info_name + "_" + annotation_name]

        # Plot the individual feature ROCs (without modeling)
        if annotation_name != "Hapestry" and truth_info_name == "TruvariBench_TP":
            cmap = pyplot.get_cmap('tab20')

            for name,index in feature_indexes.items():
                if index > 12:
                    y_trivial = x[:,index]
                    y, y_trivial = downsample_test_data(y_true, y_trivial)

                    color = cmap(float(index-12)/(len(feature_indexes)-12))

                    axes, fpr, tpr, thresholds = plot_roc_curve(y_true=y, y_predict=y_trivial, axes=axes, color=color, label=name, style=':')

        tandem_axes = plot_tandem_stratified_roc_curves(tandem_axes, records, y_true, y_predict, truth_info_name, annotation_name)
        # length_axis = plot_length_stratified_roc_curves(length_axis, records, y_true, y_predict, truth_info_name, annotation_name, tandem_only=True) # BNDs do not have lenght

        # write the filtered VCF (BEFORE DOWNSAMPLING)
        write_filtered_vcf(y_predict=y_predict, threshold=0.5, records=records, input_vcf_path=eval_vcfs[0], output_vcf_path=os.path.join(output_dir, label.replace(" ", "_") + ".vcf"))

        y_true, y_predict = downsample_test_data(y_true, y_predict)
        auc = round(roc_auc_score(y_true=y_true, y_score=y_predict), 6)
        auc_label = label + ", AUC = " + str(auc)

        # print shape of y_predict and y_true; print all values in one element
        print("y_predict shape: ", y_predict.shape)
        print("y_true shape: ", y_true.shape)
        print('y_predict y')
        print(y_predict[0], y_true[0])

        axes, fpr, tpr, thresholds = plot_roc_curve(y_true=y_true, y_predict=y_predict, axes=axes, label=auc_label, use_text=True)

        roc_data[label] = (fpr, tpr, thresholds)
    
    # Generate a legend for axes and force bottom right location
    axes.plot([0, 1], [0, 1], linestyle='--', label="Random classifier, AUC = 0.5", color="gray")
    axes.legend(loc="lower right", fontsize='small')
    fig.savefig(os.path.join(output_dir,"roc_curve_all_combos.png"), dpi=200)

    tandem_axes.plot([0, 1], [0, 1], linestyle='--', label="Random classifier, AUC = 0.5", color="gray")
    tandem_axes.legend(loc="lower right", fontsize='small')
    tandem_fig.savefig(os.path.join(output_dir,"roc_curve_tandem_stratified.png"), dpi=200)

    # The code below has been commented because BNDs do not have length
    # for label,length_axis in length_axes.items():
    #     length_axis.plot([0, 1], [0, 1], linestyle='--', label="Random classifier", color="gray")
    #     length_axis.legend(loc="lower right", fontsize='small')
    #     length_figs[label].savefig(os.path.join(output_dir,label+"_length.png"), dpi=200)

    for label,(fpr,tpr,thresholds) in roc_data.items():
        with open(os.path.join(output_dir, label.replace(" ", "_") + "_roc.csv"), "w") as f:
            f.write("threshold,fpr,tpr\n")
            for fpr_val,tpr_val,threshold in zip(fpr,tpr,thresholds):
                f.write(f"{threshold},{fpr_val},{tpr_val}\n")

    # pyplot.show()
    # pyplot.close()


if __name__ == "__main__":
    main()
