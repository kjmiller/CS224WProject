import numpy
import numpy.random
import scipy.sparse
import partial_gradient_update  
import page_rank_update 
import sys
import random

num_nodes = 2
perturbation = 0
rho = float(sys.argv[1]) * random.random()

numpy.random.seed(0)
Q_dense = numpy.random.rand(num_nodes, num_nodes)
Q_dense /= numpy.tile(numpy.dot(Q_dense, numpy.ones((num_nodes, 1))), (1, num_nodes))
print(numpy.eye(num_nodes, num_nodes) - Q_dense.T)
A = numpy.eye(num_nodes, num_nodes) - Q_dense.T
z = 100.0 * numpy.random.rand(num_nodes, 1)
Q_grad_dense = numpy.tile(numpy.dot(A, z), (1, num_nodes)).T
Q_grad_dense += perturbation * numpy.random.rand(num_nodes, num_nodes)
Q = scipy.sparse.csr_matrix(Q_dense)
p = numpy.random.rand(num_nodes, 1)
p_grad = numpy.zeros((num_nodes, 1))
p_grad[0] = rho
p /= numpy.sum(p)
p = page_rank_update.update_p(p, Q, {"page_rank_epsilon" : 1e-12})
print(p)
Q_grad = scipy.sparse.csr_matrix(Q_grad_dense)
print(Q_grad.T.dot(p))
p_grad = partial_gradient_update.update_p_grad(p, p_grad, Q, [Q_grad], 1, {"partial_gradient_update_epsilon"  : 1e-12})
print(numpy.dot(A, p_grad) - numpy.dot(A, z))
