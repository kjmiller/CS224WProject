import os
import sys
import numpy
import networkx
import synthetic_data


def load_coauthorship_feature(feature_file, node_mapper):    
    features = {}
    f = open(feature_file)         
    for line in f:
        if line[0] == '#':
            continue
                
        node1, node2, f1, f2,f3,f4,f5,f6 = line.strip().split()
        
        n1 = node_mapper[long(node1)]
        n2 = node_mapper[long(node2)]
        features[(n1, n2)] = [int(f1), int(f2), int(f3), float(f4), float(f5), int(f6)]
        
    f.close()
    
    return features

def get_edge_ij_and_feature_stack(G, feature_file_name, num_features, node_mapper):
    feature_dict = load_coauthorship_feature(feature_file_name, node_mapper)
    
    edge_ij = numpy.zeros((2, 2 * G.number_of_edges()))
    feature_stack = numpy.zeros((2 * G.number_of_edges(), num_features))
    t = 0
    for edge in G.edges():
        edge_ij[0, t] = edge[0]
        edge_ij[1, t] = edge[1]
        feature_stack[t, :] = feature_dict[edge]
        t += 1
        edge_ij[0, t] = edge[1]
        edge_ij[1, t] = edge[0]
        feature_stack[t, :] = feature_dict[edge] #feature_dict[(edge[1], edge[0])]
        t += 1    

    return (edge_ij, feature_stack, G.copy())