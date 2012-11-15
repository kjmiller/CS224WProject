import os
import sys
import numpy
import networkx
import synthetic_data

def get_edge_ij_and_feature_stack(adjlist_file_name, feature_file_name, num_features, num_nodes = None):
	G, feature_dict = synthetic_data.load_data(adjlist_file_name, feature_file_name)
	
	if num_nodes != None:
		G.remove_nodes_from(range(num_nodes, len(G.nodes())))

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

	print("Loaded %d nodes from %s"%(len(G.nodes()), adjlist_file_name))

	return (edge_ij, feature_stack, G)

def write_spn_list(spn_file_name, spn_list):
	spn_file = open(spn_file_name, "w")
	for spn in spn_list:
		spn_file.write(";".join([str(spn[0]), ",".join(map(str, spn[1])), ",".join(map(str, spn[2]))]) + "\n")

	spn_file.close()

def get_spn_list(spn_file_name):
	spn_file = open(spn_file_name, "r")
	spn_list = []
	for line in spn_file:
		stuff = line.rstrip("\n").split(";")
		spn_list.append((int(stuff[0]), map(int, stuff[1].split(",")), map(int, stuff[2].split(","))))

	spn_file.close()
	return spn_list
