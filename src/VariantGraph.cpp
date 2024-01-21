#include "VariantGraph.hpp"
#include "Sequence.hpp"

using sv_merge::reverse_complement;
using bdsg::step_handle_t;
using bdsg::path_handle_t;

#include <cmath>

using std::min;
using std::max;
using std::sort;
using std::unique;
using std::find;
using std::distance;
using std::to_string;
using std::streamsize;


// use coord_t everywhere for chr coords from <misc.h>


namespace sv_merge {

const char VariantGraph::GFA_SEPARATOR = '\t';
const char VariantGraph::GFA_NODE_CHAR = 'S';
const char VariantGraph::GFA_LINK_CHAR = 'L';
const char VariantGraph::GFA_PATH_CHAR = 'P';
const char VariantGraph::GFA_OVERLAP_FIELD = '*';
const char VariantGraph::GFA_PATH_SEPARATOR = ',';
const char VariantGraph::GFA_FWD_CHAR = '+';
const char VariantGraph::GFA_REV_CHAR = '-';
const char VariantGraph::GFA_LINE_END = '\n';
const uint64_t VariantGraph::STREAMSIZE_MAX = numeric_limits<streamsize>::max();


VariantGraph::VariantGraph(const unordered_map<string,string>& chromosomes, const unordered_map<string,vector<interval_t>>& tandem_track):
        chromosomes(chromosomes),
        n_chromosomes(chromosomes.size()),
        tandem_track(tandem_track)
{ }


void VariantGraph::sort_and_compact_positions(vector<uint32_t>& positions) {
    if (positions.size()<=1) return;
    sort(positions.begin(),positions.end());
    const auto iterator = unique(positions.begin(),positions.end());
    positions.resize(distance(positions.begin(),iterator));
}


uint16_t VariantGraph::find_closest(const string& chromosome, uint32_t position) const {
    const vector<uint32_t>& positions = chunk_first.at(chromosome);
    const auto iterator = lower_bound(positions.begin(),positions.end(),position);
    if (iterator==positions.end()) return positions.size()-1;
    const uint16_t p = distance(positions.begin(),iterator);
    if (p==0 || positions.at(p)==position) return p;
    return position-positions.at(p-1)<=positions.at(p)-position?p-1:p;
}


void VariantGraph::add_nonreference_edge(const handle_t& from, const handle_t& to, uint32_t record_id) {
    if (graph.has_edge(from,to)) {
        const edge_t edge = graph.edge_handle(from,to);
        if (!edge_to_vcf_record.contains(edge)) edge_to_vcf_record[edge]={record_id};
        else edge_to_vcf_record.at(edge).emplace_back(record_id);
        vcf_record_to_edge.at(record_id).emplace_back(edge);
    }
    else {
        graph.create_edge(from,to);
        edge_t edge = graph.edge_handle(from,to);
        edge_to_vcf_record[edge]={record_id};
        vcf_record_to_edge.at(record_id).emplace_back(edge);
    }
}


bool VariantGraph::intersect(const interval_t& interval1, const interval_t& interval2) {
    const uint32_t from1 = interval1.first;
    const uint32_t to1 = interval1.second-1;
    const uint32_t from2 = interval2.first;
    const uint32_t to2 = interval2.second-1;
    return max(from1,from2)<=min(to1,to2);
}


uint32_t VariantGraph::get_flank_boundary_right(const string& chromosome_id, uint32_t pos, uint16_t flank_length) {
    const auto CHROMOSOME_LENGTH = (uint32_t)chromosomes.at(chromosome_id).length();
    const vector<interval_t>& intervals = tandem_track.at(chromosome_id);
    const uint32_t N_INTERVALS = intervals.size();

    if (pos==CHROMOSOME_LENGTH-1) return UINT32_MAX;
    if (N_INTERVALS==0) return min(pos+flank_length,CHROMOSOME_LENGTH-1);
    tmp_interval.first=pos+1; tmp_interval.second=pos+1+flank_length;
    auto iter = std::lower_bound(intervals.begin(),intervals.end(),tmp_interval);
    uint32_t p = iter-intervals.begin();
    if (p>0 && intersect(tmp_interval,intervals.at(p-1))) {
        tmp_interval.first=intervals.at(p-1).second;
        tmp_interval.second=tmp_interval.first+flank_length;
    }
    while (p<N_INTERVALS && intersect(tmp_interval,intervals.at(p))) {
        tmp_interval.first=intervals.at(p).second;
        tmp_interval.second=tmp_interval.first+flank_length;
        p++;
    }
    return tmp_interval.first>=CHROMOSOME_LENGTH?UINT32_MAX:min((uint32_t)tmp_interval.second-1,CHROMOSOME_LENGTH-1);
}


uint32_t VariantGraph::get_flank_boundary_left(const string& chromosome_id, uint32_t pos, uint16_t flank_length) {
    const vector<interval_t>& intervals = tandem_track.at(chromosome_id);
    const uint32_t N_INTERVALS = intervals.size();

    if (pos==0) return UINT32_MAX;
    if (N_INTERVALS==0) return pos>=flank_length?pos-flank_length:0;
    tmp_interval.first=pos>=flank_length?pos-flank_length:0; tmp_interval.second=pos;
    auto iter = std::lower_bound(intervals.begin(),intervals.end(),tmp_interval);
    uint32_t p = iter-intervals.begin();
    if (p<N_INTERVALS && intersect(tmp_interval,intervals.at(p))) {
        tmp_interval.second=intervals.at(p).first;
        tmp_interval.first=tmp_interval.second>=flank_length?tmp_interval.second-flank_length:0;
    }
    if (p>0) {
        p--;
        while (intersect(tmp_interval,intervals.at(p))) {
            tmp_interval.second=intervals.at(p).first;
            tmp_interval.first=tmp_interval.second>=flank_length?tmp_interval.second-flank_length:0;
            if (p==0) break;
            else p--;
        }
    }
    return tmp_interval.second==0?UINT32_MAX:(uint32_t)tmp_interval.first;
}


void VariantGraph::build(vector<VcfRecord>& records, uint16_t flank_length, bool deallocate_ref_alt, const vector<string>& callers) {
    graph.clear();
    this->vcf_records=std::move(records);
    n_vcf_records=vcf_records.size();
    bnd_ids.clear(); bnd_ids.reserve(n_vcf_records);
    node_handles.clear(); node_handles.reserve(n_chromosomes);
    for (const auto& chromosome: chromosomes) node_handles.emplace(chromosome.first,vector<handle_t>());
    node_to_chromosome.clear(); node_to_chromosome.reserve(n_vcf_records);
    insertion_handles.clear(); insertion_handles.reserve(n_vcf_records);
    edge_to_vcf_record.clear(); edge_to_vcf_record.reserve(n_vcf_records);
    if (vcf_records.empty()) { vcf_record_to_edge.clear(); return; }

    const string CHROMOSOME_ID = vcf_records.at(0).chrom;
    const uint32_t CHROMOSOME_LENGTH = chromosomes.at(CHROMOSOME_ID).length();
    const uint8_t BIN_SIZE = 5;  // Arbitrary, just for stats.
    bool previous_handle_exists;
    uint8_t orientation_cis, orientation_trans, is_bnd_single;
    uint16_t n_positions;
    uint32_t i, j, p, q, p_prime, q_prime, first_pos, last_pos, trans_pos;
    size_t r;
    string tmp_buffer;
    pair<uint32_t, uint32_t> tmp_pair;
    handle_t previous_handle;

    // Collecting every distinct first position of a reference node, and building all non-reference nodes.
    // Remark: it can happen that the list of first positions of a chromosome contains the position right after the last
    // position of the chromosome (e.g. if there is a BND that connects the ends of two chromosomes, or if there is an
    // INS after the end of a chromosome).
    cerr << "Number of VCF records: " << n_vcf_records << '\n';
    print_vcf_records_stats(BIN_SIZE,callers);
    chunk_first_raw.clear(); chunk_first_raw.reserve(n_chromosomes);
    for (const auto& chromosome: chromosomes) chunk_first_raw.emplace(chromosome.first,vector<uint32_t>());
    vector<uint32_t>& first_positions = chunk_first_raw.at(CHROMOSOME_ID);
    for (i=0; i<n_vcf_records; i++) {
        VcfRecord& record = vcf_records.at(i);
        record.get_reference_coordinates(false,tmp_pair);
        if (tmp_pair.first==UINT32_MAX || tmp_pair.second==UINT32_MAX) continue;
        if (tmp_pair.first!=tmp_pair.second) {
            first_positions.push_back(tmp_pair.first); first_positions.push_back(tmp_pair.second);
            if (record.sv_type==VcfReader::TYPE_REPLACEMENT) {
                const handle_t handle_alt = graph.create_handle(record.alt.substr(1));
                insertion_handles.emplace_back(handle_alt);
            }
        }
        else {
            if (record.sv_type==VcfReader::TYPE_INSERTION && !record.is_symbolic) {
                first_positions.emplace_back(tmp_pair.first);
                const handle_t handle_ins = graph.create_handle(record.alt.substr(1));
                insertion_handles.emplace_back(handle_ins);
            }
            else {
                is_bnd_single=record.is_breakend_single();
                if (record.sv_type==VcfReader::TYPE_BREAKEND && is_bnd_single>=1 && !record.is_breakend_virtual(chromosomes)) {
                    // Virtual telomeric breakends, and single breakends with no inserted sequence, carry no information.
                    r=record.get_info_field(VcfReader::MATEID_STR,0,tmp_buffer);
                    if (r!=string::npos && bnd_ids.contains(tmp_buffer)) {
                        // NOP: the breakend was already processed at its mate record.
                        // Remark: If the MATEID field is missing from a mate record, the non-reference nodes created
                        // are the same.
                    }
                    else {
                        bnd_ids.emplace(record.id);
                        orientation_cis=record.get_breakend_orientation_cis();
                        first_positions.emplace_back(orientation_cis==1?tmp_pair.first+1:tmp_pair.first);
                        if (is_bnd_single==2) {
                            trans_pos=record.get_breakend_pos();
                            trans_pos--; // Zero-based
                            orientation_trans=record.get_breakend_orientation_trans();
                            record.get_breakend_chromosome(tmp_buffer);
                            chunk_first_raw[tmp_buffer].emplace_back(orientation_trans==1?trans_pos+1:trans_pos);
                            record.get_breakend_inserted_sequence(tmp_buffer);
                            if (!tmp_buffer.empty()) {
                                const handle_t handle_ins = graph.create_handle(tmp_buffer);
                                insertion_handles.emplace_back(handle_ins);
                            }
                        }
                        else {
                            record.get_breakend_inserted_sequence(tmp_buffer);
                            const handle_t handle_ins = graph.create_handle(tmp_buffer);
                            insertion_handles.emplace_back(handle_ins);
                        }
                    }
                }
            }
        }
    }
    cerr << "Table: Chromosome involved | N. distinct breakpoints\n";
    for (auto& [chromosome,first_positions]: chunk_first_raw) {
        const uint32_t n_breakpoints = first_positions.size();
        if (n_breakpoints==0) continue;
        sort_and_compact_positions(first_positions);
        cerr << chromosome << ": ";
        for (const auto& position: first_positions) cerr << std::to_string(position) << ',';
        cerr << '\n';
        cerr << chromosome << ',' << first_positions.size() << '\n';
    }
    cerr << "Number of non-reference nodes: " << insertion_handles.size() << '\n';

    // Building all reference nodes and all reference edges, taking into account the tandem repeat track and the
    // flanking requirement.
    // Remark: this procedure works also when the list of first positions of a chromosome contains the position right
    // after the last position of the chromosome.
    chunk_first.clear(); chunk_first.reserve(n_chromosomes);
    for (const auto& [chromosome,first_positions]: chunk_first_raw) {
        n_positions=first_positions.size();
        if (n_positions==0) continue;
        const string& chrom_sequence = chromosomes.at(chromosome);
        const uint32_t chrom_length = chrom_sequence.size();
        vector<uint32_t> first_positions_new;
        first_positions_new.reserve(n_positions);
        vector<handle_t>& handles = node_handles.at(chromosome);
        handles.clear();
        p=first_positions.at(0);
        first_pos=get_flank_boundary_left(chromosome,p,flank_length);
        if (first_pos==UINT32_MAX) first_pos=0;
        if (p!=first_pos) {
            const handle_t reference_handle = graph.create_handle(chrom_sequence.substr(first_pos,p-first_pos));
            handles.emplace_back(reference_handle);
            first_positions_new.emplace_back(first_pos);
            node_to_chromosome[graph.get_id(reference_handle)]=pair<string,uint32_t>(chromosome,first_pos);
            previous_handle_exists=true; previous_handle=reference_handle;
        }
        else previous_handle_exists=false;
        for (j=1; j<n_positions; j++) {
            p=first_positions.at(j-1);
            p_prime=get_flank_boundary_right(chromosome,p,flank_length);
            q=first_positions.at(j);
            q_prime=get_flank_boundary_left(chromosome,q,flank_length);
            if (p_prime!=UINT32_MAX && q_prime!=UINT32_MAX && p_prime<q_prime) {
                const handle_t reference_handle_1 = graph.create_handle(chrom_sequence.substr(p,p_prime-p+1));
                const handle_t reference_handle_2 = graph.create_handle(chrom_sequence.substr(q_prime,q-q_prime));
                handles.emplace_back(reference_handle_1);
                handles.emplace_back(reference_handle_2);
                first_positions_new.emplace_back(p);
                first_positions_new.emplace_back(q_prime);
                node_to_chromosome[graph.get_id(reference_handle_1)]=pair<string,uint32_t>(chromosome,p);
                node_to_chromosome[graph.get_id(reference_handle_2)]=pair<string,uint32_t>(chromosome,q_prime);
                if (previous_handle_exists) graph.create_edge(previous_handle,reference_handle_1);
                previous_handle_exists=true; previous_handle=reference_handle_2;
            }
            else {
                const handle_t reference_handle = graph.create_handle(chrom_sequence.substr(p,first_positions.at(j)-p));
                handles.emplace_back(reference_handle);
                first_positions_new.emplace_back(p);
                node_to_chromosome[graph.get_id(reference_handle)]=pair<string,uint32_t>(chromosome,p);
                if (previous_handle_exists) graph.create_edge(previous_handle,reference_handle);
                previous_handle_exists=true; previous_handle=reference_handle;
            }
        }
        p=first_positions.at(n_positions-1);
        first_positions_new.emplace_back(p);
        if (p<chrom_length) {
            last_pos=get_flank_boundary_right(chromosome,p,flank_length);
            if (last_pos==UINT32_MAX) last_pos=chrom_length-1;
            const handle_t reference_handle = graph.create_handle(chrom_sequence.substr(p,last_pos+1-p));
            handles.emplace_back(reference_handle);
            node_to_chromosome[graph.get_id(reference_handle)]=pair<string,uint32_t>(chromosome,p);
            if (previous_handle_exists) graph.create_edge(previous_handle,reference_handle);
        }
        chunk_first.emplace(chromosome,first_positions_new);
    }
    chunk_first_raw.clear();
    cerr << "Number of reference nodes: " << to_string(graph.get_node_count()-insertion_handles.size()) << '\n';
    cerr << "Total number of nodes: " << to_string(graph.get_node_count()) << '\n';
    cerr << "Table: Chromosome involved | First position of every reference node\n";
    for (const auto& [chromosome,first_positions]: chunk_first) {
        const uint32_t n_breakpoints = first_positions.size();
        if (n_breakpoints==0) continue;
        cerr << chromosome << ": ";
        for (const auto& position: first_positions) cerr << std::to_string(position) << ',';
        cerr << '\n';
    }
    cerr << "Table: Chromosome involved | N. reference nodes\n";
    for (const auto& [chromosome,chunks]: node_handles) {
        const uint32_t n_chunks = chunks.size();
        if (n_chunks>0) cerr << chromosome << ',' << n_chunks << '\n';
    }
    const uint32_t n_reference_edges = graph.get_edge_count();
    cerr << "Number of reference edges: " << n_reference_edges << '\n';

    // Building all non-reference edges
    vcf_record_to_edge.reserve(n_vcf_records);
    for (i=0; i<vcf_record_to_edge.size(); i++) vcf_record_to_edge.at(i).clear();
    for (i=vcf_record_to_edge.size(); i<n_vcf_records; i++) vcf_record_to_edge.emplace_back();
    const vector<handle_t>& handles = node_handles.at(CHROMOSOME_ID);
    bnd_ids.clear(); j=0;
    for (i=0; i<n_vcf_records; i++) {
        VcfRecord& record = vcf_records.at(i);
        record.get_reference_coordinates(false,tmp_pair);
        if (tmp_pair.first==UINT32_MAX || tmp_pair.second==UINT32_MAX) continue;
        if (tmp_pair.first!=tmp_pair.second) {
            p=find_closest(CHROMOSOME_ID,tmp_pair.first);
            q=find_closest(CHROMOSOME_ID,tmp_pair.second);
            if (record.sv_type==VcfReader::TYPE_DELETION) {
                if (p!=0 && tmp_pair.second!=CHROMOSOME_LENGTH) add_nonreference_edge(handles.at(p-1), handles.at(q), i);
            }
            else if (record.sv_type==VcfReader::TYPE_DUPLICATION || record.sv_type==VcfReader::TYPE_CNV) add_nonreference_edge(handles.at(q-1),handles.at(p),i);
            else if (record.sv_type==VcfReader::TYPE_INVERSION) {
                if (p!=0) add_nonreference_edge(handles.at(p-1), graph.flip(handles.at(q-1)), i);
                if (tmp_pair.second!=CHROMOSOME_LENGTH) add_nonreference_edge(graph.flip(handles.at(p)),handles.at(q),i);
            }
            else if (record.sv_type==VcfReader::TYPE_REPLACEMENT) {
                const handle_t& insertion_handle = insertion_handles.at(j);
                if (p!=0) add_nonreference_edge(handles.at(p-1),insertion_handle,i);
                if (tmp_pair.second!=CHROMOSOME_LENGTH) add_nonreference_edge(insertion_handle,handles.at(q),i);
                j++;
            }
        }
        else {
            if (record.sv_type==VcfReader::TYPE_INSERTION && !record.is_symbolic) {
                p=find_closest(CHROMOSOME_ID,tmp_pair.first);
                const handle_t& insertion_handle = insertion_handles.at(j);
                if (p!=0) add_nonreference_edge(handles.at(p-1),insertion_handle,i);
                if (tmp_pair.first!=CHROMOSOME_LENGTH) add_nonreference_edge(insertion_handle,handles.at(p),i);
                j++;
            }
            else {
                is_bnd_single=record.is_breakend_single();
                if (record.sv_type==VcfReader::TYPE_BREAKEND && is_bnd_single>=1 && !record.is_breakend_virtual(chromosomes)) {
                    // Virtual telomeric breakends, and single breakends with no inserted sequence, carry no information.
                    r=record.get_info_field(VcfReader::MATEID_STR,0,tmp_buffer);
                    if (r!=string::npos && bnd_ids.contains(tmp_buffer)) {
                        // NOP: the breakend was already processed at its mate record.
                        // Remark: If the MATEID field is missing from a mate record, the non-reference edges created
                        // are the same.
                    }
                    else {
                        bnd_ids.emplace(record.id);
                        orientation_cis=record.get_breakend_orientation_cis();
                        const handle_t& handle_from = orientation_cis==1?handles.at(find_closest(CHROMOSOME_ID,tmp_pair.first+1)-1):graph.flip(handles.at(find_closest(CHROMOSOME_ID,tmp_pair.first)));
                        if (is_bnd_single==2) {
                            record.get_breakend_chromosome(tmp_buffer);
                            trans_pos=record.get_breakend_pos()-1;
                            orientation_trans=record.get_breakend_orientation_trans();
                            const handle_t& handle_to = orientation_trans==1?graph.flip(node_handles.at(tmp_buffer).at(find_closest(tmp_buffer,trans_pos+1)-1)):node_handles.at(tmp_buffer).at(find_closest(tmp_buffer,trans_pos));
                            record.get_breakend_inserted_sequence(tmp_buffer);
                            if (tmp_buffer.empty()) add_nonreference_edge(handle_from,handle_to,i);
                            else {
                                const handle_t& insertion_handle = orientation_cis==1?insertion_handles.at(j):graph.flip(insertion_handles.at(j));
                                add_nonreference_edge(handle_from,insertion_handle,i);
                                add_nonreference_edge(insertion_handle,handle_to,i);
                                j++;
                            }
                        }
                        else {
                            record.get_breakend_inserted_sequence(tmp_buffer);
                            const handle_t& insertion_handle = orientation_cis==1?insertion_handles.at(j):graph.flip(insertion_handles.at(j));
                            add_nonreference_edge(handle_from,insertion_handle,i);
                            j++;
                        }
                    }
                }
            }
        }
    }
    cerr << "Total number of edges: " << to_string(graph.get_edge_count()) << '\n';
    cerr << "Number of non-reference edges: " << to_string(graph.get_edge_count()-n_reference_edges) << '\n';
    cerr << "Number of chrA-chrB edges: " << to_string(get_n_interchromosomal_edges()) << '\n';
    cerr << "Number of chr-INS edges: " << to_string(get_n_ins_edges()) << '\n';
    cerr << "Number of nodes with edges on just one side: " << to_string(get_n_dangling_nodes()) << '\n';
    unordered_set<nid_t> dangling_nodes;
    get_dangling_nodes(false,dangling_nodes);
    print_edge_histograms();

    // Clearing temporary space
    chunk_first.clear(); node_handles.clear(); insertion_handles.clear(); bnd_ids.clear();

    // Deallocating REF and ALT
    if (deallocate_ref_alt) {
        for (auto& record: vcf_records) { record.ref.clear(); record.alt.clear(); }
    }
}


void VariantGraph::to_gfa(const path& gfa_path) const {
    ofstream outstream;

    outstream.open(gfa_path.string());
    graph.for_each_handle([&](handle_t handle) { outstream << GFA_NODE_CHAR << GFA_SEPARATOR << graph.get_id(handle) << GFA_SEPARATOR << graph.get_sequence(handle) << GFA_LINE_END; });
    graph.for_each_edge([&](edge_t edge) { outstream << GFA_LINK_CHAR << GFA_SEPARATOR << graph.get_id(edge.first) << GFA_SEPARATOR << (graph.get_is_reverse(edge.first)?GFA_REV_CHAR:GFA_FWD_CHAR) << GFA_SEPARATOR << graph.get_id(edge.second) << GFA_SEPARATOR << (graph.get_is_reverse(edge.second)?GFA_REV_CHAR:GFA_FWD_CHAR) << GFA_SEPARATOR << GFA_OVERLAP_FIELD << GFA_LINE_END; });
    graph.for_each_path_handle([&](path_handle_t path) {
        outstream << GFA_PATH_CHAR << GFA_SEPARATOR << graph.get_path_name(path) << GFA_SEPARATOR;
        const step_handle_t source = graph.path_begin(path);
        const step_handle_t destination = graph.path_back(path);
        outstream << graph.get_id(graph.get_handle_of_step(source)) << (graph.get_is_reverse(graph.get_handle_of_step(source))?GFA_REV_CHAR:GFA_FWD_CHAR);
        step_handle_t from = source;
        while (graph.has_next_step(from)) {
            const step_handle_t to = graph.get_next_step(from);
            if (to==destination) break;
            outstream << GFA_PATH_SEPARATOR << graph.get_id(graph.get_handle_of_step(from)) << (graph.get_is_reverse(graph.get_handle_of_step(from))?GFA_REV_CHAR:GFA_FWD_CHAR);
            from=to;
        }
        outstream << GFA_SEPARATOR << GFA_OVERLAP_FIELD << GFA_LINE_END;
    });
    outstream.close();
}


vector<string> VariantGraph::load_gfa(const path& gfa_path) {
    bool is_node, is_link, is_path, orientation_from, orientation_to;
    char c;
    uint8_t n_fields;
    uint32_t id_from, id_to;
    string buffer, node_sequence, path_name, path_encoding;
    vector<string> node_ids;
    ifstream instream;

    // Loading node IDs
    instream.open(gfa_path.string());
    if (!instream.is_open() || !instream.good()) throw runtime_error("ERROR: could not read file: "+gfa_path.string());
    n_fields=0;
    while (instream.get(c)) {
        if (c!=GFA_SEPARATOR && c!=GFA_LINE_END) { buffer.push_back(c); continue; }
        n_fields++;
        if (n_fields==1) {
            if (buffer.at(0)!=GFA_NODE_CHAR) {
                instream.ignore(STREAMSIZE_MAX,GFA_LINE_END);
                n_fields=0;
            }
        }
        else if (n_fields==2) node_ids.emplace_back(buffer);
        else {  // No other field is used
            if (c!=GFA_LINE_END) instream.ignore(STREAMSIZE_MAX,GFA_LINE_END);
            n_fields=0;
        }
        buffer.clear();
    }
    instream.close();
    if (node_ids.size()>1) sort(node_ids.begin(),node_ids.end());

    // Loading graph
    destroy_paths(); graph.clear();
    instream.open(gfa_path.string());
    is_node=false; is_link=false; is_path=false; orientation_from=false; orientation_to=false; id_from=UINT32_MAX; id_to=UINT32_MAX;
    n_fields=0;
    while (instream.get(c)) {
        if (c!=GFA_SEPARATOR && c!=GFA_LINE_END) { buffer.push_back(c); continue; }
        n_fields++;
        if (n_fields==1) {
            is_node=buffer.at(0)==GFA_NODE_CHAR;
            is_link=buffer.at(0)==GFA_LINK_CHAR;
            is_path=buffer.at(0)==GFA_PATH_CHAR;
            if (!is_node && !is_link && !is_path) {
                instream.ignore(STREAMSIZE_MAX,GFA_LINE_END);
                buffer.clear(); n_fields=0;
                continue;
            }
        }
        else if (n_fields==2) {
            if (is_path) path_name=buffer;
            else {
                const auto iterator = lower_bound(node_ids.begin(),node_ids.end(),buffer);
                id_from=distance(node_ids.begin(),iterator)+1;  // Node IDs cannot be zero
            }
        }
        else if (n_fields==3) {
            if (is_node) node_sequence=buffer;
            else if (is_link) orientation_from=buffer.at(0)==GFA_FWD_CHAR;
            else if (is_path) path_encoding=buffer;
        }
        else if (n_fields==4) {
            if (is_link) {
                const auto iterator = lower_bound(node_ids.begin(),node_ids.end(),buffer);
                id_to=distance(node_ids.begin(),iterator)+1;  // Node IDs cannot be zero
            }
        }
        else if (n_fields==5) {
            if (is_link) orientation_to=buffer.at(0)==GFA_FWD_CHAR;
        }
        else {  /* No other field is used */ }
        if (c==GFA_LINE_END) {
            if (is_node) graph.create_handle(node_sequence,id_from);
            else if (is_link) graph.create_edge(orientation_from?graph.get_handle(id_from):graph.flip(graph.get_handle(id_from)),orientation_to?graph.get_handle(id_to):graph.flip(graph.get_handle(id_to)));
            else if (is_path) load_gfa_path(path_encoding,node_ids,path_name,buffer);
            n_fields=0;
        }
        buffer.clear();
    }
    instream.close();
    cerr << "Loaded " << to_string(graph.get_node_count()) << " nodes, " << to_string(graph.get_edge_count()) << " edges, " << to_string(graph.get_path_count()) << " paths from the GFA file.\n";
    return node_ids;
}


void VariantGraph::load_gfa_path(const string& path_encoding, const vector<string>& node_ids, const string& path_name, string& buffer) {
    const uint16_t LENGTH = path_encoding.length();
    char c;
    uint16_t i;
    uint16_t node_id;

    path_handle_t path = graph.create_path_handle(path_name);
    buffer.clear();
    for (i=0; i<LENGTH; i++) {
        c=path_encoding.at(i);
        if (c==GFA_PATH_SEPARATOR) continue;
        else if (c!=GFA_FWD_CHAR && c!=GFA_REV_CHAR) { buffer.push_back(c); continue; }
        const auto iterator = lower_bound(node_ids.begin(),node_ids.end(),buffer);
        node_id=distance(node_ids.begin(),iterator)+1;  // Node IDs cannot be zero
        graph.append_step(path,c==GFA_FWD_CHAR?graph.get_handle(node_id):graph.flip(graph.get_handle(node_id)));
        buffer.clear();
    }
}


void VariantGraph::destroy_paths() {
    vector<path_handle_t> path_handles;
    graph.for_each_path_handle([&](const path_handle_t& path){ path_handles.push_back(path); });
    for (auto& path_handle: path_handles) graph.destroy_path(path_handle);
}


void VariantGraph::mark_edge(const edge_t& query, const vector<edge_t>& edges, vector<uint8_t>& flags) const {
    const uint8_t N_EDGES = edges.size();
    const edge_t reverse_query(graph.flip(query.second),graph.flip(query.first));
    edge_t edge;

    for (uint8_t i=0; i<N_EDGES; i++) {
        edge=edges.at(i);
        if (edge==query) {
            if (flags.at(i)==2) flags.at(i)=3;
            else flags.at(i)=1;
            return;
        }
        else if (edge==reverse_query) {
            if (flags.at(i)==1) flags.at(i)=3;
            else flags.at(i)=2;
            return;
        }
    }
}


void VariantGraph::print_supported_vcf_records(ofstream& outstream, bool print_all_records, const vector<string>& callers) {
    const uint8_t N_CALLERS = callers.size();
    uint32_t i;
    uint8_t j, n_edges, n_forward, n_reverse, n_both;
    vector<vector<uint32_t>> caller_count;

    printed.clear(); printed.reserve(n_vcf_records);
    if (print_all_records) {
        for (i=0; i<n_vcf_records; i++) printed.emplace_back(true);
    }
    else {
        flags.reserve(n_vcf_records);
        for (i=flags.size(); i<n_vcf_records; i++) flags.emplace_back();
        for (i=0; i<n_vcf_records; i++) {
            printed.emplace_back(false);
            flags.at(i).reserve(vcf_record_to_edge.at(i).size());
        }
        graph.for_each_path_handle([&](path_handle_t path) {
            for (i=0; i<n_vcf_records; i++) {
                if (printed.at(i)) continue;
                n_edges=vcf_record_to_edge.at(i).size();
                flags.at(i).clear();
                for (j=0; j<n_edges; j++) flags.at(i).emplace_back(0);
            }
            step_handle_t from = graph.path_begin(path);  // `graph` breaks circular paths
            const step_handle_t last = graph.path_back(path);  // `graph` breaks circular paths
            while (from!=last) {
                const step_handle_t to = graph.get_next_step(from);
                const edge_t edge = graph.edge_handle(graph.get_handle_of_step(from),graph.get_handle_of_step(to));
                if (edge_to_vcf_record.contains(edge)) {
                    for (const uint32_t& k: edge_to_vcf_record.at(edge)) mark_edge(edge,vcf_record_to_edge.at(k),flags.at(k));
                }
                from=to;
            }
            for (i=0; i<n_vcf_records; i++) {
                if (printed.at(i)) continue;
                n_edges=vcf_record_to_edge.at(i).size();
                if (n_edges==0)  continue;  // VCF records not used to build the graph have no supporting edge
                n_forward=0; n_reverse=0; n_both=0;
                for (j=0; j<n_edges; j++) {
                    switch (flags.at(i).at(j)) {
                        case 1: n_forward++; break;
                        case 2: n_reverse++; break;
                        case 3: n_both++; break;
                    }
                }
                if (n_forward+n_both==n_edges || n_reverse+n_both==n_edges) printed.at(i)=true;
            }
        });
    }
    for (i=0; i<N_CALLERS; i++) caller_count.emplace_back(vector<uint32_t> {0,0,0,0,0});
    for (i=0; i<n_vcf_records; i++) {
        if (!printed.at(i)) continue;
        VcfRecord& record = vcf_records.at(i);
        record.print(outstream); outstream << '\n';
        if (!callers.empty()) {
            if (record.sv_type==VcfReader::TYPE_INSERTION) increment_caller_count(record,callers,caller_count,0);
            else if (record.sv_type==VcfReader::TYPE_DUPLICATION) increment_caller_count(record,callers,caller_count,1);
            else if (record.sv_type==VcfReader::TYPE_DELETION) increment_caller_count(record,callers,caller_count,2);
            else if (record.sv_type==VcfReader::TYPE_INVERSION) increment_caller_count(record,callers,caller_count,3);
            else if (record.sv_type==VcfReader::TYPE_BREAKEND) increment_caller_count(record,callers,caller_count,4);
        }
    }
    if (!callers.empty()) {
        cerr << "Histogram: Caller | #INS printed | #DUP printed | #DEL printed | #INV printed | #BND printed\n";
        for (i=0; i<caller_count.size(); i++) {
            cerr << callers.at(i);
            for (j=0; j<caller_count.at(i).size(); j++) cerr << ',' << to_string(caller_count.at(i).at(j));
            cerr << '\n';
        }
    }
}


void VariantGraph::paths_to_vcf_records(ofstream& outstream) {
    const string DEFAULT_QUAL = "60";  // Arbitrary
    const string DEFAULT_FILTER = VcfReader::PASS_STR;
    const string DEFAULT_FORMAT = "GT";
    const string DEFAULT_SAMPLE = "0/1";
    const string SUFFIX_REPLACEMENT = VcfReader::VCF_SEPARATOR+DEFAULT_QUAL+VcfReader::VCF_SEPARATOR+DEFAULT_FILTER+VcfReader::VCF_SEPARATOR+VcfReader::VCF_MISSING_CHAR+VcfReader::VCF_SEPARATOR+DEFAULT_FORMAT+VcfReader::VCF_SEPARATOR+DEFAULT_SAMPLE+'\n';
    const string SUFFIX_BND = VcfReader::VCF_SEPARATOR+DEFAULT_QUAL+VcfReader::VCF_SEPARATOR+DEFAULT_FILTER+VcfReader::VCF_SEPARATOR+"SVTYPE=BND"+VcfReader::VCF_SEPARATOR+DEFAULT_FORMAT+VcfReader::VCF_SEPARATOR+DEFAULT_SAMPLE+'\n';
    char pos_base;
    uint32_t pos;
    string old_sequence, new_sequence, alt_sequence;

    graph.for_each_path_handle([&](path_handle_t path) {
        const step_handle_t source = graph.path_begin(path);  // `graph` breaks circular paths
        const step_handle_t destination = graph.path_back(path);  // `graph` breaks circular paths
        const handle_t source_handle = graph.get_handle_of_step(source);
        const handle_t destination_handle = graph.get_handle_of_step(destination);
        const nid_t id1 = graph.get_id(source_handle);
        const nid_t id2 = graph.get_id(destination_handle);
        if (node_to_chromosome.contains(id1) && node_to_chromosome.contains(id2)) {
            const pair<string,uint32_t> source_coordinate = node_to_chromosome.at(id1);
            const pair<string,uint32_t> destination_coordinate = node_to_chromosome.at(id2);
            const bool is_source_reversed = graph.get_is_reverse(source_handle);
            const bool is_destination_reversed = graph.get_is_reverse(destination_handle);
            new_sequence.clear();
            if ( source_coordinate.first==destination_coordinate.first &&
                 ( (!is_source_reversed && !is_destination_reversed && source_coordinate.second<destination_coordinate.second) ||
                   (is_source_reversed && is_destination_reversed && source_coordinate.second>destination_coordinate.second)
                   )
                 ) {
                // Source and destination are on the same chromosome and in the same orientation. We represent this as a
                // replacement record.
                if (!is_source_reversed) {
                    pos=source_coordinate.second+graph.get_length(source_handle);  // One-based
                    pos_base=chromosomes.at(source_coordinate.first).at(pos-1);
                    old_sequence=chromosomes.at(source_coordinate.first).substr(pos,destination_coordinate.second-pos);
                    step_handle_t from = source;
                    while (graph.has_next_step(from)) {
                        const step_handle_t to = graph.get_next_step(from);
                        if (to==destination) break;
                        new_sequence.append(graph.get_sequence(graph.get_handle_of_step(to)));
                        from=to;
                    }
                }
                else {
                    pos=destination_coordinate.second+graph.get_length(destination_handle);  // One-based
                    pos_base=chromosomes.at(destination_coordinate.first).at(pos-1);
                    old_sequence=chromosomes.at(destination_coordinate.first).substr(pos,source_coordinate.second-pos);
                    step_handle_t from = destination;
                    while (graph.has_previous_step(from)) {
                        const step_handle_t to = graph.get_previous_step(from);
                        if (to==source) break;
                        new_sequence.append(graph.get_sequence(graph.flip(graph.get_handle_of_step(to))));
                        from=to;
                    }
                }
                outstream << source_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << old_sequence << VcfReader::VCF_SEPARATOR << pos_base << new_sequence << SUFFIX_REPLACEMENT;
            }
            else {
                // Source and destination are not on the same chromosome, or not in the same orientation. We represent
                // this as a BND record with inserted sequence.
                if (!is_source_reversed) {
                    pos=source_coordinate.second+graph.get_length(source_handle);  // One-based
                    pos_base=chromosomes.at(source_coordinate.first).at(pos-1);
                    step_handle_t from = source;
                    while (graph.has_next_step(from)) {
                        const step_handle_t to = graph.get_next_step(from);
                        if (to==destination) break;
                        new_sequence.append(graph.get_sequence(graph.get_handle_of_step(to)));
                        from=to;
                    }
                    alt_sequence.clear(); alt_sequence.append(pos_base+new_sequence);
                    if (!is_destination_reversed) {
                        alt_sequence.append(VcfReader::BREAKEND_CHAR_OPEN+destination_coordinate.first);
                        alt_sequence.append(VcfReader::BREAKEND_SEPARATOR+to_string(destination_coordinate.second+1)+VcfReader::BREAKEND_CHAR_OPEN);
                    }
                    else {
                        alt_sequence.append(VcfReader::BREAKEND_CHAR_CLOSE+destination_coordinate.first);
                        alt_sequence.append(VcfReader::BREAKEND_SEPARATOR+to_string(destination_coordinate.second+graph.get_length(destination_handle))+VcfReader::BREAKEND_CHAR_CLOSE);
                    }
                    outstream << source_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << VcfReader::VCF_SEPARATOR << alt_sequence << SUFFIX_BND;
                }
                else {
                    pos=source_coordinate.second+1;  // One-based
                    pos_base=chromosomes.at(source_coordinate.first).at(pos-1);
                    step_handle_t from = source;
                    while (graph.has_next_step(from)) {
                        const step_handle_t to = graph.get_next_step(from);
                        if (to==destination) break;
                        new_sequence.append(graph.get_sequence(graph.get_handle_of_step(to)));
                        from=to;
                    }
                    reverse_complement(new_sequence);
                    alt_sequence.clear();
                    if (!is_destination_reversed) {
                        alt_sequence.append(VcfReader::BREAKEND_CHAR_OPEN+destination_coordinate.first);
                        alt_sequence.append(VcfReader::BREAKEND_SEPARATOR+to_string(destination_coordinate.second+1)+VcfReader::BREAKEND_CHAR_OPEN);
                    }
                    else {
                        alt_sequence.append(VcfReader::BREAKEND_CHAR_CLOSE+destination_coordinate.first);
                        alt_sequence.append(VcfReader::BREAKEND_SEPARATOR+to_string(destination_coordinate.second+graph.get_length(destination_handle))+VcfReader::BREAKEND_CHAR_CLOSE);
                    }
                    alt_sequence.append(new_sequence+pos_base);
                    outstream << source_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << VcfReader::VCF_SEPARATOR << alt_sequence << SUFFIX_BND;
                }
            }
        }
        else if (node_to_chromosome.contains(id1)) {
            // We represent this as a "single BND" record with inserted sequence.
            const pair<string,uint32_t> source_coordinate = node_to_chromosome.at(id1);
            const bool is_source_reversed = graph.get_is_reverse(source_handle);
            new_sequence.clear();
            if (!is_source_reversed) {
                pos=source_coordinate.second+graph.get_length(source_handle);  // One-based
                pos_base=chromosomes.at(source_coordinate.first).at(pos-1);
                step_handle_t from = source;
                while (graph.has_next_step(from)) {
                    const step_handle_t to = graph.get_next_step(from);
                    new_sequence.append(graph.get_sequence(graph.get_handle_of_step(to)));
                    if (to==destination) break;
                    from=to;
                }
                outstream << source_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << VcfReader::VCF_SEPARATOR << pos_base << new_sequence << VcfReader::VCF_MISSING_CHAR << SUFFIX_BND;
            }
            else {
                pos=source_coordinate.second+1;  // One-based
                pos_base=chromosomes.at(source_coordinate.first).at(pos-1);
                step_handle_t from = source;
                while (graph.has_next_step(from)) {
                    const step_handle_t to = graph.get_next_step(from);
                    new_sequence.append(graph.get_sequence(graph.get_handle_of_step(to)));
                    if (to==destination) break;
                    from=to;
                }
                reverse_complement(new_sequence);
                outstream << source_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << VcfReader::VCF_SEPARATOR << VcfReader::VCF_MISSING_CHAR << new_sequence << pos_base << SUFFIX_BND;
            }
        }
        else if (node_to_chromosome.contains(id2)) {
            // We represent this as a "single BND" record with inserted sequence.
            const pair<string,uint32_t> destination_coordinate = node_to_chromosome.at(id2);
            const bool is_destination_reversed = graph.get_is_reverse(destination_handle);
            new_sequence.clear();
            if (!is_destination_reversed) {
                pos=destination_coordinate.second+1;  // One-based
                pos_base=chromosomes.at(destination_coordinate.first).at(pos-1);
                step_handle_t from = destination;
                while (graph.has_previous_step(from)) {
                    const step_handle_t to = graph.get_previous_step(from);
                    new_sequence.append(graph.get_sequence(graph.flip(graph.get_handle_of_step(to))));
                    if (to==source) break;
                    from=to;
                }
                reverse_complement(new_sequence);
                outstream << destination_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << VcfReader::VCF_SEPARATOR << VcfReader::VCF_MISSING_CHAR << new_sequence << pos_base << SUFFIX_BND;
            }
            else {
                pos=destination_coordinate.second+graph.get_length(destination_handle);  // One-based
                pos_base=chromosomes.at(destination_coordinate.first).at(pos-1);
                step_handle_t from = destination;
                while (graph.has_previous_step(from)) {
                    const step_handle_t to = graph.get_previous_step(from);
                    new_sequence.append(graph.get_sequence(graph.flip(graph.get_handle_of_step(to))));
                    if (to==source) break;
                    from=to;
                }
                outstream << destination_coordinate.first << VcfReader::VCF_SEPARATOR << pos << VcfReader::VCF_SEPARATOR << graph.get_path_name(path) << VcfReader::VCF_SEPARATOR << pos_base << VcfReader::VCF_SEPARATOR << pos_base << new_sequence << VcfReader::VCF_MISSING_CHAR << SUFFIX_BND;
            }
        }
        else { /* NOP: impossible to decide chr and pos. */ }
    });
}


void VariantGraph::get_dangling_nodes(bool mode, unordered_set<nid_t>& out) const {
    if (!mode) out.clear();
    graph.for_each_handle([&](handle_t node) {
        const uint16_t left_degree = graph.get_degree(node,true);
        const uint16_t right_degree = graph.get_degree(node,false);
        if (left_degree==0 || right_degree==0) {
            if (mode) out.emplace(graph.get_id(node));
            else cerr << "Dangling node: " << to_string(graph.get_id(node)) << ":" << to_string(graph.get_is_reverse(node)) << "=" << graph.get_sequence(node) << " left_degree: " << to_string(left_degree) << " right_degree: " << to_string(right_degree) << '\n';
        }
    });
}


uint32_t VariantGraph::get_n_dangling_nodes() const {
    uint32_t out = 0;
    graph.for_each_handle([&](handle_t node) {
        if (graph.get_degree(node,true)==0 || graph.get_degree(node,false)==0) out++;
    });
    return out;
}


uint32_t VariantGraph::get_n_interchromosomal_edges() const {
    uint32_t out = 0;
    graph.for_each_edge([&](edge_t edge) {
        const nid_t node1 = graph.get_id(edge.first);
        const nid_t node2 = graph.get_id(edge.second);
        if (node_to_chromosome.contains(node1) && node_to_chromosome.contains(node2) && node_to_chromosome.at(node1).first!=node_to_chromosome.at(node2).first) out++;
    });
    return out;
}


uint32_t VariantGraph::get_n_ins_edges() const {
    uint32_t out = 0;
    graph.for_each_edge([&](edge_t edge) {
        const nid_t node1 = graph.get_id(edge.first);
        const nid_t node2 = graph.get_id(edge.second);
        if (node_to_chromosome.contains(node1)!=node_to_chromosome.contains(node2)) out++;
    });
    return out;
}


void VariantGraph::print_edge_histograms() const {
    const char SEPARATOR = ',';
    uint32_t n_edges = edge_to_vcf_record.size();
    if (n_edges==0) return;

    uint32_t i, last;
    vector<uint32_t> histogram;

    for (i=0; i<=n_vcf_records; i++) histogram.emplace_back(0);
    for (const auto& [key,value]: edge_to_vcf_record) histogram.at(value.size())++;
    cerr << "Histogram: X = n. VCF records | Y = n. non-reference edges supported by X VCF records\n";
    cerr << "Remark: Only VCF records in input to graph construction are counted.\n";
    for (last=n_vcf_records; last>=1; last--) {
        if (histogram.at(last)>0) break;
    }
    for (i=0; i<=last; i++) cerr << to_string(i) << SEPARATOR << to_string(histogram.at(i)) << '\n';

    histogram.reserve(n_edges+1);
    for (i=0; i<histogram.size(); i++) histogram.at(i)=0;
    for (i=histogram.size(); i<=n_edges; i++) histogram.emplace_back(0);
    for (i=0; i<n_vcf_records; i++) histogram.at(vcf_record_to_edge.at(i).size())++;
    cerr << "Histogram: X = n. non-reference edges | Y = n. VCF records creating X non-reference edges\n";
    cerr << "Remark: Only VCF records in input to graph construction are counted.\n";
    for (last=n_edges-1; last>=1; last--) { if (histogram.at(last)>0) break; }
    for (i=0; i<=last; i++) cerr << to_string(i) << SEPARATOR << to_string(histogram.at(i)) << '\n';

    n_edges=graph.get_edge_count();
    histogram.reserve(n_edges+1);
    for (i=0; i<histogram.size(); i++) histogram.at(i)=0;
    for (i=histogram.size(); i<=n_edges; i++) histogram.emplace_back(0);
    graph.for_each_handle([&](handle_t node) { histogram.at(max(graph.get_degree(node,false),graph.get_degree(node,true)))++; });
    cerr << "Histogram: X = max degree of a node | Y = n. nodes with max degree X\n";
    for (last=n_edges-1; last>=1; last--) { if (histogram.at(last)>0) break; }
    for (i=0; i<=last; i++) cerr << to_string(i) << SEPARATOR << to_string(histogram.at(i)) << '\n';
}


void VariantGraph::print_vcf_records_stats(uint8_t bin_size, const vector<string>& callers) const {
    const char SEPARATOR = ',';
    const uint8_t N_CALLERS = callers.size();
    uint8_t i, j;
    vector<uint32_t> lengths_ins, lengths_dup, lengths_del, lengths_inv;
    vector<vector<uint32_t>> caller_count;
    vector<pair<string,string>> bnds;
    string buffer;

    for (i=0; i<N_CALLERS; i++) caller_count.emplace_back(vector<uint32_t> {0,0,0,0,0,});
    for (const auto& record: vcf_records) {
        if (record.sv_type==VcfReader::TYPE_INSERTION) {
            lengths_ins.emplace_back(record.sv_length/bin_size);
            if (!callers.empty()) increment_caller_count(record,callers,caller_count,0);
        }
        else if (record.sv_type==VcfReader::TYPE_DUPLICATION) {
            lengths_dup.emplace_back(record.sv_length/bin_size);
            if (!callers.empty()) increment_caller_count(record,callers,caller_count,1);
        }
        else if (record.sv_type==VcfReader::TYPE_DELETION) {
            lengths_del.emplace_back(record.sv_length/bin_size);
            if (!callers.empty()) increment_caller_count(record,callers,caller_count,2);
        }
        else if (record.sv_type==VcfReader::TYPE_INVERSION) {
            lengths_inv.emplace_back(record.sv_length/bin_size);
            if (!callers.empty()) increment_caller_count(record,callers,caller_count,3);
        }
        else if (record.sv_type==VcfReader::TYPE_BREAKEND) {
            record.get_breakend_chromosome(buffer);
            if (record.chrom<buffer) bnds.emplace_back(record.chrom,buffer);
            else bnds.emplace_back(buffer,record.chrom);
            if (!callers.empty()) increment_caller_count(record,callers,caller_count,4);
        }
    }
    cerr << "Histogram: SV type | SV length (binned by " << to_string(bin_size) << "bp) | Number of occurrences\n";
    print_sv_lengths(lengths_ins,bin_size,VcfReader::INS_STR);
    print_sv_lengths(lengths_dup,bin_size,VcfReader::DUP_STR);
    print_sv_lengths(lengths_del,bin_size,VcfReader::DEL_STR);
    print_sv_lengths(lengths_inv,bin_size,VcfReader::INV_STR);
    cerr << "Histogram: BND | chrA | chrB | Number of occurrences\n";
    print_bnds(bnds);
    if (!callers.empty()) {
        cerr << "Histogram: Caller | #INS | #DUP | #DEL | #INV | #BND\n";
        for (i=0; i<caller_count.size(); i++) {
            cerr << callers.at(i);
            for (j=0; j<caller_count.at(i).size(); j++) cerr << SEPARATOR << to_string(caller_count.at(i).at(j));
            cerr << '\n';
        }
    }
}


void VariantGraph::print_sv_lengths(vector<uint32_t>& lengths, uint8_t bin_size, const string& prefix) {
    const char SEPARATOR = ',';
    uint32_t i, previous, previous_count;
    const uint32_t SIZE = lengths.size();

    if (SIZE==0) return;
    if (SIZE>1) sort(lengths.begin(),lengths.end());
    previous=lengths.at(0); previous_count=1;
    for (i=1; i<SIZE; i++) {
        if (lengths.at(i)!=previous) {
            cerr << prefix << SEPARATOR << to_string(previous*bin_size) << SEPARATOR << to_string(previous_count) << '\n';
            previous=lengths.at(i); previous_count=1;
        }
        else previous_count++;
    }
    cerr << prefix << SEPARATOR << to_string(previous*bin_size) << SEPARATOR << to_string(previous_count) << '\n';
}


void VariantGraph::print_bnds(vector<pair<string,string>>& bnds) {
    const char SEPARATOR = ',';
    const uint32_t SIZE = bnds.size();
    uint32_t i, previous_count;

    if (SIZE==0) return;
    if (SIZE>1) sort(bnds.begin(),bnds.end());
    pair<string,string>& previous = bnds.at(0);
    previous_count=1;
    for (i=1; i<SIZE; i++) {
        if (bnds.at(i)!=previous) {
            cerr << "BND" << SEPARATOR << previous.first << SEPARATOR << previous.second << SEPARATOR << to_string(previous_count) << '\n';
            previous=bnds.at(i); previous_count=1;
        }
        else previous_count++;
    }
    cerr << "BND" << SEPARATOR << previous.first << SEPARATOR << previous.second << SEPARATOR << to_string(previous_count) << '\n';
}


void VariantGraph::increment_caller_count(const VcfRecord& record, const vector<string>& callers, vector<vector<uint32_t>>& caller_count, uint8_t column) {
    const uint8_t N_CALLERS = callers.size();
    string lower = record.id;
    lowercase_string(lower);
    for (uint8_t i=0; i<N_CALLERS; i++) {
        if (lower.find(callers.at(i))!=string::npos) { caller_count.at(i).at(column)++; return; }
    }
}


void VariantGraph::print_graph_signature(uint16_t max_steps, const path& output_path) const {
    const char SEPARATOR = ',';
    uint32_t i;
    string buffer;
    vector<string> signatures, tokens;

    graph.for_each_handle([&](handle_t handle) {
        tokens.clear();
        print_signature_impl(handle,graph.get_sequence(handle),0,max_steps,tokens);
        const handle_t rev_handle = graph.flip(handle);
        print_signature_impl(rev_handle,graph.get_sequence(rev_handle),0,max_steps,tokens);
        sort(tokens.begin(),tokens.end());
        buffer.clear();
        for (i=0; i<tokens.size(); i++) { buffer.append(tokens.at(i)); buffer.push_back(SEPARATOR); }
        signatures.emplace_back(buffer);
    });
    sort(signatures.begin(),signatures.end());
    ofstream output_file(output_path.string());
    for (i=0; i<signatures.size(); i++) output_file << signatures.at(i) << '\n';
    output_file.close();
}


void VariantGraph::print_signature_impl(const handle_t& handle, const string& path, uint16_t steps_performed, uint16_t max_steps, vector<string>& strings) const {
    if (steps_performed==max_steps || graph.get_degree(handle,false)==0) {
        strings.emplace_back(path);
        return;
    }
    graph.follow_edges(handle,false,[&](handle_t neighbor_handle) {
        string new_path = path;
        new_path.append(graph.get_sequence(neighbor_handle));
        print_signature_impl(neighbor_handle,new_path,steps_performed+1,max_steps,strings);
    });
}


void VariantGraph::load_edge_record_map(const vector<pair<edge_t,uint32_t>>& map, uint32_t n_vcf_records) {
    edge_to_vcf_record.clear(); vcf_record_to_edge.clear();
    vcf_record_to_edge.reserve(n_vcf_records);
    for (uint32_t i=0; i<n_vcf_records; i++) vcf_record_to_edge.emplace_back();
    for (const auto& pair: map) {
        if (edge_to_vcf_record.contains(pair.first)) edge_to_vcf_record.at(pair.first).emplace_back(pair.second);
        else edge_to_vcf_record[pair.first]={pair.second};
        vcf_record_to_edge.at(pair.second).emplace_back(pair.first);
    }
}


void VariantGraph::node_to_chromosome_clear() { node_to_chromosome.clear(); }


void VariantGraph::node_to_chromosome_insert(const string& node_label, const vector<string> node_labels, const string& chromosome, uint32_t position) {
    const auto iterator = lower_bound(node_labels.begin(),node_labels.end(),node_label);
    const uint32_t node_id = distance(node_labels.begin(),iterator)+1;  // Node IDs cannot be zero
    node_to_chromosome[node_id]=pair<string,uint32_t>(chromosome,position);
}

}