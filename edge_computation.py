import os
import sys
import numpy
import scipy.sparse

def logistic(z):
	return 1.0 / (1.0 + numpy.exp(-1.0 * z))

def compute_A(w, feature_stack, edge_ij, edge_strength_fun, num_nodes):
	A_data = edge_strength_fun(numpy.dot(feature_stack, w)).flatten()
	#print(A_data)
	A = scipy.sparse.csr_matrix((A_data, edge_ij), shape = (num_nodes, num_nodes))
	return (A, A_data)

def compute_Q(A, A_data, edge_ij, s, num_nodes, params):
	#print("compute_Q")
	alpha = params["alpha"]
	row_sums = A.sum(axis = 1)
	Q_data = numpy.zeros(edge_ij.shape[1])
	for t in range(edge_ij.shape[1]):
		i = edge_ij[0, t]
		j = edge_ij[1, t]
		Q_data[t] = A_data[t] / (1.0 * row_sums[i])
			
	#print(Q_dense)
	Q_global = scipy.sparse.csr_matrix((Q_data, edge_ij), shape = (num_nodes, num_nodes))
	ij_alpha = numpy.vstack((numpy.reshape(numpy.array(range(num_nodes)), (1, num_nodes)), s * numpy.ones((1, num_nodes))))
	alpha_part = scipy.sparse.csr_matrix((alpha * numpy.ones(num_nodes), ij_alpha), shape = (num_nodes, num_nodes))
	Q = (1 - alpha) * Q_global + alpha_part
	#print(Q)
	#print("done")
	return Q

def logistic_grad(f):
	return f * (1 - f)

def compute_df_dwk(k, feature_stack, A_data, edge_ij, edge_strength_grad_fun, num_nodes, params):
	df_dwk_data = feature_stack[:, k] * edge_strength_grad_fun(A_data)
	df_dwk = scipy.sparse.csr_matrix((df_dwk_data,edge_ij), shape = (num_nodes, num_nodes))
	return (df_dwk, df_dwk_data)

def compute_dQ_dwk(k, df_dwk, df_dwk_data, A, params, edge_ij, A_data):
	#print("compute_dQ_dwk")
	alpha = params["alpha"]
	dQ_dwk_data = numpy.zeros(edge_ij.shape[1])
	row_sums_A = A.sum(axis = 1)
	denom = numpy.square(row_sums_A)
	row_sums_A_grad = df_dwk.sum(axis = 1)
	for t in range(edge_ij.shape[1]):
		i = edge_ij[0, t]
		j = edge_ij[1, t]
		dQ_dwk_data[t] = (1 - alpha) * (df_dwk_data[t] * row_sums_A[i] - A_data[t]  * row_sums_A_grad[i]) / denom[i]

	dQ_dwk = scipy.sparse.csr_matrix((dQ_dwk_data, edge_ij), shape = df_dwk.shape)
	#print(dQ_dwk.todense())
	#print("done")
	return dQ_dwk
