import os
import sys
import numpy
import networkx
import synthetic_data

def get_edge_ij_and_feature_stack(adjlist_file_name, feature_file_name, num_features):
	G, feature_dict = synthetic_data.load_data(adjlist_file_name, feature_file_name)
	edge_ij = numpy.zeros((2, 2 * len(G.edges())))
	feature_stack = numpy.zeros((2 * len(G.edges()), num_features))
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

	return (edge_ij, feature_stack)

def get_snp_list(snp_file_name):
	snp_file = open(snp_file_name, "r")
	snp_list = []
	for line in snp_file:
		stuff = line.rstrip("\n").split(";")
		snp_list.append((int(stuff[0]), map(int, stuff[1].split(",")), map(int, stuff[2].split(","))))

	snp_file.close()
	return snp_list
