#pragma once

#include "HeteroGraph.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <string>
#include <vector>

using std::unordered_map;
using std::unordered_set;
using std::to_string;
using std::vector;
using std::string;
using std::pair;

#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

using operations_research::sat::CpModelBuilder;
using operations_research::Domain;
using operations_research::sat::IntVar;
using operations_research::sat::BoolVar;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::LinearExpr;


namespace sv_merge {


class TransMap {
    /// Attributes
    HeteroGraph<HeteroNode> graph;

    // Pains me to add yet another map but here it is
    unordered_map<int64_t,string> sequences;

    const string sample_node_name;
    const string read_node_name;
    const string path_node_name;

public:
    TransMap();

    /// Building
    void add_sample(const string& name);
    void add_read(const string& name);
    void add_read(const string& name, const string& sequence);
    void add_path(const string& name);
    void add_edge(const string& a, const string& b);
    void add_edge(const string& a, const string& b, float weight);
    void construct_optimizer();

    /// Accessing
    const HeteroNode& get_node(int64_t id) const;

    void for_each_sample(const function<void(const string& name, int64_t id)>& f) const;
    void for_each_read(const function<void(const string& name, int64_t id)>& f) const;
    void for_each_path(const function<void(const string& name, int64_t id)>& f) const;

    void get_read_sample(const string& read_name, string& result) const;
    void get_read_sample(int64_t read_id, string& result) const;

    void for_each_read_of_sample(const string& sample_name, const function<void(const string& name, int64_t id)>& f) const;
    void for_each_read_of_sample(int64_t sample_id, const function<void(int64_t read_id)>& f) const;
    void for_each_read_of_path(int64_t path_id, const function<void(int64_t id)>& f) const;

    void for_each_sample_of_read(const string& read_name, const function<void(const string& name, int64_t id)>& f) const;
    void for_each_sample_of_path(const string& path_name, const function<void(const string& name, int64_t id)>& f) const;

    void for_each_path_of_sample(const string& sample_name, const function<void(const string& name, int64_t id)>& f) const;

    void for_each_path_of_read(int64_t read_id, const function<void(int64_t path_id)>& f) const;

    void for_each_read_to_path_edge(const function<void(int64_t read_id, int64_t path_id, float weight)>& f) const;
};


TransMap::TransMap():
    sample_node_name("sample_node"),
    read_node_name("read_node"),
    path_node_name("path_node")
{
    graph.add_node(sample_node_name, 'S');
    graph.add_node(read_node_name, 'R');
    graph.add_node(path_node_name, 'P');
}


const HeteroNode& TransMap::get_node(int64_t id) const{
    return graph.get_node(id);
}


void TransMap::add_read(const string& name){
    if (name == read_node_name or name == sample_node_name or name == path_node_name){
        throw runtime_error("ERROR: cannot add node with preset node name: " + name);
    }
    graph.add_node(name, 'R');
    graph.add_edge(read_node_name, name, 0);
}


void TransMap::add_read(const string& name, const string& sequence){
    if (name == read_node_name or name == sample_node_name or name == path_node_name){
        throw runtime_error("ERROR: cannot add node with preset node name: " + name);
    }
    graph.add_node(name, 'R');
    graph.add_edge(read_node_name, name, 0);

    sequences.emplace(graph.name_to_id(name), sequence);
}


void TransMap::add_sample(const string& name){
    if (name == read_node_name or name == sample_node_name or name == path_node_name){
        throw runtime_error("ERROR: cannot add node with preset node name: " + name);
    }
    graph.add_node(name, 'S');
    graph.add_edge(sample_node_name, name, 0);
}


void TransMap::add_path(const string& name){
    if (name == read_node_name or name == sample_node_name or name == path_node_name){
        throw runtime_error("ERROR: cannot add node with preset node name: " + name);
    }
    graph.add_node(name, 'P');
    graph.add_edge(path_node_name, name, 0);
}


void TransMap::add_edge(const string& a, const string& b){
    graph.add_edge(a,b,0);
}


void TransMap::add_edge(const string& a, const string& b, float weight){
    graph.add_edge(a,b,weight);
}


void TransMap::get_read_sample(const string& read_name, string& result) const{
    result.clear();

    // Using the HeteroGraph back end means that we iterate, even for a 1:1 mapping.
    graph.for_each_neighbor_of_type(read_name, 'S', [&](const HeteroNode& neighbor, int64_t id){
        // Check for cases that should be impossible
        if (not result.empty()){
            throw runtime_error("ERROR: multiple samples found for read: " + read_name);
        }

        result = neighbor.name;
    });

    if (result.empty()){
        throw runtime_error("ERROR: no sample found for read: " + read_name);
    }
}


void TransMap::get_read_sample(int64_t read_id, string& result) const{
    result.clear();

    if (graph.get_node(read_id).type != 'R'){
        throw runtime_error("ERROR: non-read ID provided for get_read_sample: " + to_string(read_id) + " " + graph.get_node(read_id).name);
    }

    // Using the HeteroGraph back end means that we iterate, even for a 1:1 mapping.
    graph.for_each_neighbor_of_type(read_id, 'S', [&](int64_t id){
        // Check for cases that should be impossible
        if (not result.empty()){
            throw runtime_error("ERROR: multiple samples found for id: " + to_string(read_id));
        }

        result = graph.get_node(id).name;
    });

    // For this project, every read must have exactly one sample.
    if (result.empty()){
        throw runtime_error("ERROR: no sample found for id: " + to_string(read_id));
    }
}


void TransMap::for_each_read(const function<void(const string& name, int64_t id)>& f) const{
    graph.for_each_neighbor_of_type(read_node_name, 'R', [&](const HeteroNode& neighbor, int64_t id){
        f(neighbor.name, id);
    });
}


void TransMap::for_each_sample(const function<void(const string& name, int64_t id)>& f) const{
    graph.for_each_neighbor_of_type(sample_node_name, 'S', [&](const HeteroNode& neighbor, int64_t id){
        f(neighbor.name, id);
    });
}


void TransMap::for_each_path(const function<void(const string& name, int64_t id)>& f) const{
    graph.for_each_neighbor_of_type(path_node_name, 'P', [&](const HeteroNode& neighbor, int64_t id){
        f(neighbor.name, id);
    });
}


void TransMap::for_each_read_of_sample(const string& sample_name, const function<void(const string& name, int64_t id)>& f) const{
    graph.for_each_neighbor_of_type(sample_name, 'R', [&](const HeteroNode& neighbor, int64_t id){
        f(neighbor.name, id);
    });
}


void TransMap::for_each_read_of_sample(int64_t sample_id, const function<void(int64_t read_id)>& f) const{
    graph.for_each_neighbor_of_type(sample_id, 'R', [&](int64_t id){
        f(id);
    });
}


void TransMap::for_each_path_of_read(int64_t read_id, const function<void(int64_t path_id)>& f) const{
    graph.for_each_neighbor_of_type(read_id, 'P', [&](int64_t id){
        f(id);
    });
}


void TransMap::for_each_read_of_path(int64_t path_id, const function<void(int64_t id)>& f) const{
    graph.for_each_neighbor_of_type(path_id, 'R', [&](int64_t r){
        f(r);
    });
}


void TransMap::for_each_sample_of_read(const string& read_name, const function<void(const string& name, int64_t id)>& f) const{
    graph.for_each_neighbor_of_type(read_name, 'S', [&](const HeteroNode& neighbor, int64_t id){
        f(neighbor.name, id);
    });
}


void TransMap::for_each_sample_of_path(const string& path_name, const function<void(const string& name, int64_t id)>& f) const{
    auto id = graph.name_to_id(path_name);
    graph.for_each_neighbor_of_type(id, 'R', [&](int64_t r){
        graph.for_each_neighbor_of_type(r, 'S', [&](int64_t s){
            f(graph.get_node(s).name, s);
        });
    });
}


void TransMap::for_each_path_of_sample(const string& sample_name, const function<void(const string& name, int64_t id)>& f) const{
    auto id = graph.name_to_id(sample_name);
    graph.for_each_neighbor_of_type(id, 'R', [&](int64_t r){
        graph.for_each_neighbor_of_type(r, 'P', [&](int64_t p){
            f(graph.get_node(p).name, p);
        });
    });
}


void TransMap::for_each_read_to_path_edge(const function<void(int64_t read_id, int64_t path_id, float weight)>& f) const{
    auto id = graph.name_to_id(read_node_name);

    // Starting from the source node which connects to all reads, find all neighbors (read nodes)
    graph.for_each_neighbor_of_type(id, 'R', [&](int64_t read_id){
        // Find all path-type neighbors
        graph.for_each_neighbor_of_type(read_id, 'P', [&](int64_t path_id, float w){
            f(read_id, path_id, w);
        });
    });
}


}
