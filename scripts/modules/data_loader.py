from torch.utils.data.dataset import Dataset
from collections import defaultdict
import numpy as np
import torch
import sys
import time

import vcfpy


def get_type_index(record: vcfpy.Record):
    if record.ALT[0].type == "INS":
        return 0
    elif record.ALT[0].type == "DEL":
        return 1
    elif record.ALT[0].type == "DUP":
        return 2
    elif record.ALT[0].type == "INV":
        return 3
    else:
        return 4


type_names = ["INS","DEL","DUP","NA"]

def feature_length_verification(x, feature_names):
    if len(x) == 0: # No items were added yet
        print('ERROR: No vector "x" has not elements.')
    elif len(feature_names) != len(x[-1]):
        print(feature_names)
        print("ERROR: feature names and data length mismatch: names:%d x:%d" % (len(feature_names), len(x[-1])))
    added_feature_status = True
    return added_feature_status


def load_features_from_vcf(
        records: list,
        x: list,
        y: list,
        feature_names: list,
        vcf_path: str,
        truth_info_name: str,
        annotation_name: str,
        data_conditions : dict,
        filter_fn=None,
        contigs=None,
        evaluate_complex_bnds=False,
        ):
    # For a given record, the features are in the following order: [BND HAPESTRY_READS, Mate of BND HAPESTRY_READS]
    bnd = dict() # bnd = {ID/MATEID : HAPESTRY_READS + [is_true] + [chrom] + [alt]}

    evaluate_complex_bnds = evaluate_complex_bnds

    all_bnds = data_conditions['all_bnds']
    concatenate_bnds = data_conditions['concatenate_bnds']
    max_bnds = data_conditions['max_bnds']
    sum_bnds = data_conditions['sum_bnds']
    average_bnds = data_conditions['average_bnds']


    print(vcf_path)
    reader = vcfpy.Reader.from_path(vcf_path)

    type_vector = [0,0,0,0,0]
    added_feature_status = False
    for record in reader:
        id = record.ID[0]
        chrom = record.CHROM[3:]
        info = record.INFO
        
        alt = record.ALT[0].__dict__ # Obtain 'ALT' column parameters
        alt_type = type(record.ALT[0]).__name__
   
        hapestry_reads = list(map(float,info["HAPESTRY_READS"]))

        if filter_fn is not None:
            if not(filter_fn(record)):
                continue

        if contigs is not None:
            if record.CHROM not in contigs:
                continue

        if record.calls is None:
            exit("ERROR: no calls in record: " + record.ID)
        elif len(record.calls) != 1:
            exit("ERROR: multiple calls in record: " + record.ID)

        if info['SVTYPE'] != alt['type']:
            if info['SVTYPE'] == 'INS':
                if alt['type'] != 'INDEL':
                    print(f'Warning: BND type in INFO={info["SVTYPE"]} does not match with ALT={alt["type"]}.')

            elif info['SVTYPE'] == 'DEL':
                if alt['type'] != 'SYMBOLIC':
                    print(f'Warning: BND type in INFO={info["SVTYPE"]} does not match with ALT={alt["type"]}.')

            elif info['SVTYPE'] == 'DUP':
                if alt['type'] != 'SYMBOLIC':
                    print(f'Warning: BND type in INFO={info["SVTYPE"]} does not match with ALT={alt["type"]}.')

            elif info['SVTYPE'] == 'INV':
                if alt['type'] != 'SYMBOLIC':
                    print(f'Warning: BND type in INFO={info["SVTYPE"]} does not match with ALT={alt["type"]}.')

            else:
                print(f'Warning: Unknown BND type in INFO={info["SVTYPE"]} paired with ALT={alt["type"]}')

        if all_bnds == True:
            if evaluate_complex_bnds == True:
                if alt_type != 'BreakEnd':
                    continue
        else:
            if alt_type != 'BreakEnd':
                    continue

        if truth_info_name.lower() == "hapestry":
            is_true = float(info["HAPESTRY_REF_MAX"]) > 0.9
        elif truth_info_name is not None:
            if truth_info_name not in info:
                sys.stderr.write("ERROR: truth info not found in record: " + str(record.ID) + "\n")
                continue
            is_true = info[truth_info_name]
        else:
            is_true = 0

        # Record does not have a MATEID. Concatenate ID data and set zeros for MATEID data. The following part is for simple events and single BNDs
        if alt_type != 'BreakEnd':

            # Append a list that will be populated with data
            x.append([])
            # Make the missing record as an array of zeros
            if 'mate_of_' not in id:
                if concatenate_bnds:
                    # Add the hapestry features for the ID records
                    x[-1].extend(hapestry_reads)
                    # Add a list of zeros for the mate records
                    x[-1].extend([0] * len(hapestry_reads))
                elif max_bnds or sum_bnds:
                    x[-1].extend(hapestry_reads)
                elif average_bnds:
                    x[-1].extend([(e1 + e2) / 2 for e1, e2 in zip(hapestry_reads, [0]*len(hapestry_reads))])
            else:
                if concatenate_bnds:
                    # Add a list of zeros for the ID records
                    x[-1].extend([0] * len(hapestry_reads))
                    # Add the hapestry features for the mate records
                    x[-1].extend(hapestry_reads)
                elif max_bnds or sum_bnds:
                    x[-1].extend(hapestry_reads)
                elif average_bnds:
                    x[-1].extend([(e1 + e2) / 2 for e1, e2 in zip(hapestry_reads, [0]*len(hapestry_reads))])

            x[-1].extend([0,0,1]) # Add BND_Connection information (single breakend)

            
            if alt_type == 'SingleBreakEnd':
                # Single BNDs
                if alt['mate_orientation'] == None:
                    if alt['orientation'] == '+':
                        x[-1].extend([1,0,1,0])
                    elif alt['orientation'] == '-':
                        x[-1].extend([1,0,0,0])
                    else:
                        print(f'Error: Unexpected "orientation={alt["orientation"]}" for a single BND. Record ID: {id}')
                elif alt['mate_orientation'] == '+' or alt['mate_orientation'] == '-':
                    print(f'Error: Unexpected to see a "mate_orientation={alt["mate_orientation"]}" for a single BND. Considering the BND as single BND. Record ID: {id}')
                    # Ignoring mate_orientation, and considering it as a single BND
                    if alt['orientation'] == '+':
                        x[-1].extend([1,0,1,0])
                    elif alt['orientation'] == '-':
                        x[-1].extend([1,0,0,0])
                    else:
                        print(f'Error: Unexpected "orientation={alt["orientation"]}" for a single BND. Record ID: {id}')
                else:
                    print(f'Error: Unexpected "mate_orientation"={alt["mate_orientation"]} for a single BND. Record ID: {id}')
            else:
                # For simple events
                x[-1].extend([0,0,0,0])
            y.append(is_true)
            records.append(record)
            if added_feature_status == False:
                feature_names.extend(["hapestry_data_" + str(i) for i in range(2*len(hapestry_reads)+len([0]*7) if concatenate_bnds else len(hapestry_reads)+len([0]*7) if max_bnds or sum_bnds or average_bnds else None)])
                added_feature_status = feature_length_verification(x, feature_names)
            continue

        mate_id = info['MATEID'][0]

        if 'mate_of_' not in id:
            # Concatenate data if MATEID was found before
            if mate_id not in bnd:
                # If mate was not found before, save data until it is found
                bnd[id] = hapestry_reads + [is_true] + [chrom] + [alt] # Concatenate the lists
                
            else:
                alt = bnd[mate_id].pop(-1)

                # Obtain the chrom pair no.
                chrom_pair = bnd[mate_id].pop(-1)

                # Obtain the maximum of the is_true values of both, ID and MATEID:
                is_true_max = max(bnd[mate_id].pop(-1), is_true)

                # Append a list that will be populated with data
                x.append([])

                if concatenate_bnds:
                    # Populate the list with data
                    x[-1].extend(hapestry_reads)
                    x[-1].extend(bnd[mate_id])
                elif max_bnds:
                    x[-1].extend(list(np.maximum(np.array(hapestry_reads), np.array(bnd[mate_id]))))
                elif sum_bnds:
                    x[-1].extend([sum(x) for x in zip(hapestry_reads, bnd[mate_id])])
                elif average_bnds:
                    x[-1].extend([(e1 + e2) / 2 for e1, e2 in zip(hapestry_reads, bnd[mate_id])])

                if chrom == chrom_pair:
                    x[-1].extend([1,0,0]) # Add BND_Connection information (intra breakend)
                
                else:
                    x[-1].extend([0,1,0]) # Add BND_Connection information (inter breakend)

                if alt['mate_orientation'] == None:
                    # This part should account for Telomeres
                    if alt['orientation'] == '+':
                        x[-1].extend([1,0,1,0])
                    elif alt['orientation'] == '-':
                        x[-1].extend([1,0,0,0])
                    else:
                        print(f'Error: Unexpected "orientation={alt["orientation"]}" for a BND. Record ID: {id}')
                elif alt['mate_orientation'] == '+' or alt['mate_orientation'] == '-':
                    if alt['orientation'] == '+' and alt['mate_orientation'] == '+':
                        x[-1].extend([1,1,1,1])
                    elif alt['orientation'] == '+' and alt['mate_orientation'] == '-':
                        x[-1].extend([1,1,1,0])
                    elif alt['orientation'] == '-' and alt['mate_orientation'] == '+':
                        x[-1].extend([1,1,0,1])
                    elif alt['orientation'] == '-' and alt['mate_orientation'] == '-':
                        x[-1].extend([1,1,0,0])
                    else:
                        print(f'Error: Unexpected "orientation={alt["orientation"]}" for a BND. Record ID: {id}')
                else:
                    print(f'Error: Unexpected "mate_orientation"={alt["mate_orientation"]} for a BND. Record ID: {id}')
                y.append(is_true_max)
                records.append(record)
                if added_feature_status == False:
                    feature_names.extend(["hapestry_data_" + str(i) for i in range(len(hapestry_reads + bnd[mate_id])+len([0]*7) if concatenate_bnds else len(hapestry_reads)+len([0]*7) if max_bnds or sum_bnds or average_bnds else None)])
                    added_feature_status = feature_length_verification(x, feature_names)
                del bnd[mate_id] # remove to save RAM


        else:
            # Concatenate data if ID exists previously. 'mate_id' does not have "mate_of_" due to previous 'if statement' 
            if mate_id not in bnd:
                # If ID  was not found before, save data until it is found
                bnd[id] = hapestry_reads + [is_true] + [chrom] + [alt] # Concatenate the lists
            else:
                alt = bnd[mate_id].pop(-1)

                # Obtain the chrom pair no.
                chrom_pair = bnd[mate_id].pop(-1)

                # Obtain the maximum of the is_true values of both, ID and MATEID:
                is_true_max = max(bnd[mate_id].pop(-1), is_true)

                # Append a list that will be populated with data
                x.append([])
                if concatenate_bnds:
                    # Populate the list with data
                    x[-1].extend(bnd[mate_id])
                    x[-1].extend(hapestry_reads)
                elif max_bnds:
                    x[-1].extend(list(np.maximum(np.array(hapestry_reads), np.array(bnd[mate_id]))))
                elif sum_bnds:
                    x[-1].extend([sum(x) for x in zip(hapestry_reads, bnd[mate_id])])
                elif average_bnds:
                    x[-1].extend([(e1 + e2) / 2 for e1, e2 in zip(hapestry_reads, bnd[mate_id])])

                if chrom == chrom_pair:
                    x[-1].extend([1,0,0]) # Add BND_Connection information (intra breakend)
                
                else:
                    x[-1].extend([0,1,0]) # Add BND_Connection information (inter breakend)

                if alt['mate_orientation'] == None:
                    # This part should account for Telomeres
                    if alt['orientation'] == '+':
                        x[-1].extend([1,0,1,0])
                    elif alt['orientation'] == '-':
                        x[-1].extend([1,0,0,0])
                    else:
                        print(f'Error: Unexpected "orientation={alt["orientation"]}" for a BND. Record ID: {id}')
                elif alt['mate_orientation'] == '+' or alt['mate_orientation'] == '-':
                    if alt['orientation'] == '+' and alt['mate_orientation'] == '+':
                        x[-1].extend([1,1,1,1])
                    elif alt['orientation'] == '+' and alt['mate_orientation'] == '-':
                        x[-1].extend([1,1,1,0])
                    elif alt['orientation'] == '-' and alt['mate_orientation'] == '+':
                        x[-1].extend([1,1,0,1])
                    elif alt['orientation'] == '-' and alt['mate_orientation'] == '-':
                        x[-1].extend([1,1,0,0])
                    else:
                        print(f'Error: Unexpected "orientation={alt["orientation"]}" for a BND. Record ID: {id}')
                else:
                    print(f'Error: Unexpected "mate_orientation"={alt["mate_orientation"]} for a BND. Record ID: {id}')
                
                y.append(is_true_max)
                records.append(record)
                if added_feature_status == False:
                    feature_names.extend(["hapestry_data_" + str(i) for i in range(len(bnd[mate_id] + hapestry_reads)+len([0]*7) if concatenate_bnds else len(hapestry_reads)+len([0]*7) if max_bnds or sum_bnds or average_bnds else None)])
                    added_feature_status = feature_length_verification(x, feature_names)        
                del bnd[mate_id] # remove to save RAM

    # 'identifier' is the key, and 'data' is the value
    for indentifier, data in bnd.items():
        alt = data.pop(-1)

        # Obtain the chrom pair no.
        chrom_pair = data.pop(-1)

        # ID as identifier
        # Obtain the is_true value of ID:
        is_true = data.pop(-1)

        # Append a list that will be populated with data
        x.append([])

        if concatenate_bnds:
            # Add a list of zeros for the ID records
            x[-1].extend(data)
            
            # Add the hapestry features for the mate records
            x[-1].extend(data) 
        elif max_bnds:
            x[-1].extend(data)
        elif sum_bnds:
            x[-1].extend([(e1 + e2) for e1, e2 in zip(data, data)])
        elif average_bnds:
            x[-1].extend([(e1 + e2) / 2 for e1, e2 in zip(data, data)])

        x[-1].extend([1,0,0]) # Add BND_Connection information (intra breakend)

        # We are considering any BNDs that do not have mates as copies of the found BNDs
        if alt['mate_orientation'] == None:
            # This part should account for Telomeres
            if alt['orientation'] == '+':
                x[-1].extend([1,0,1,0])
            elif alt['orientation'] == '-':
                x[-1].extend([1,0,0,0])
            else:
                print(f'Error: Unexpected "orientation={alt["orientation"]}" for a BND, but mate was not found. Record ID: {indentifier}')
        elif alt['mate_orientation'] == '+' or alt['mate_orientation'] == '-':
            print(f'Warning: Seen a "mate_orientation={alt["mate_orientation"]}" for a BND, but mate was not found. Considering the mate of BND as a copy of the found BND. Record ID: {indentifier}')

            if alt['orientation'] == '+' and alt['mate_orientation'] == '+':
                x[-1].extend([1,1,1,1])
            elif alt['orientation'] == '+' and alt['mate_orientation'] == '-':
                x[-1].extend([1,1,1,0])
            elif alt['orientation'] == '-' and alt['mate_orientation'] == '+':
                x[-1].extend([1,1,0,1])
            elif alt['orientation'] == '-' and alt['mate_orientation'] == '-':
                x[-1].extend([1,1,0,0])
            else:
                print(f'Error: Unexpected "orientation={alt["orientation"]}" for a BND, but mate was not found. Record ID: {indentifier}')
        else:
            print(f'Error: Unexpected "mate_orientation={alt["mate_orientation"]}" for a BND, but mate was not found. Record ID: {indentifier}')
        
        y.append(is_true)
        records.append(record)
        if added_feature_status == False:
            feature_names.extend(["hapestry_data_" + str(i) for i in range(2*len(data)+len([0]*7) if concatenate_bnds else len(data)+len([0]*7) if max_bnds or sum_bnds or average_bnds else None)])
            added_feature_status = feature_length_verification(x, feature_names)     


class VcfDatasetBNDs(Dataset):
    def __init__(self, 
                 vcf_paths: list, 
                 truth_info_name, 
                 annotation_name, 
                 data_conditions,
                 filter_fn=None, 
                 contigs=None, 
                 evaluate_complex_bnds=False, 
                 ):
        x = list()
        y = list()

        self.feature_indexes = None
        self.records = list()

        for p,path in enumerate(vcf_paths):
            if len(path) == 0 or path is None:
                continue

            feature_names = list()

            load_features_from_vcf(
                records=self.records,
                x=x,
                y=y,
                vcf_path=path,
                feature_names=feature_names,
                truth_info_name=truth_info_name,
                annotation_name=annotation_name,
                filter_fn=filter_fn,
                contigs=contigs,
                evaluate_complex_bnds=evaluate_complex_bnds,
                data_conditions=data_conditions,
            )

            self.feature_indexes = {feature_names[i]: i for i in range(len(feature_names))}

        x = np.array(x)
        y = np.array(y)

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

        self.x_data = torch.nn.functional.normalize(self.x_data, dim=0)
        self.x_data += 1e-12

        self.filter_fn = filter_fn
        self.evaluate_complex_bnds = evaluate_complex_bnds
        self.data_conditions = data_conditions

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

