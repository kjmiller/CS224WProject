import numpy.random
import numpy

num_nodes = 10
num_edges = 20
num_features = 7


edge_ij = numpy.random.randint(0, high = num_nodes, size = (2, num_edges))
feature_stack = numpy.random.rand(num_edges, num_features)


