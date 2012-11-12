import os
import sys
import numpy
import scipy.sparse

def compute_A(w, feature_stack, edge_ij, edge_strength_fun, num_nodes):
	A_data = edge_strength_fun(numpy.dot(feature_stack, w)).flatten()
	#print(A_data)
	A = scipy.sparse.csr_matrix((A_data, edge_ij), shape = (num_nodes, num_nodes))
	return (A, A_data)

def compute_Q(A, A_data, edge_ij, s, num_nodes, params):
	alpha = params["alpha"]
	row_sums = A.sum(axis = 1)
	Q_dense = numpy.zeros(A.shape)
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			Q_dense[i, j] = A[i, j] / (1.0 * row_sums[i])
			
	#print(Q_dense)
	Q_global = scipy.sparse.csr_matrix(Q_dense)
	ij_alpha = numpy.vstack((numpy.reshape(numpy.array(range(num_nodes)), (1, num_nodes)), s * numpy.ones((1, num_nodes))))
	alpha_part = scipy.sparse.csr_matrix((alpha * numpy.ones(num_nodes), ij_alpha), shape = (num_nodes, num_nodes))
	Q = (1 - alpha) * Q_global + alpha_part
	return Q

def logistic_grad(f):
	return f * (1 - f)

def compute_df_dwk(k, feature_stack, A_data, edge_ij, edge_strength_grad_fun, num_nodes, params):
	df_dwk_data = feature_stack[:, k] * edge_strength_grad_fun(A_data)
	df_dwk = scipy.sparse.csr_matrix((df_dwk_data,edge_ij), shape = (num_nodes, num_nodes))
	return df_dwk

def compute_dQ_dwk(k, df_dwk, A, params):
	alpha = params["alpha"]
	dQ_dwk_dense = numpy.zeros(df_dwk.shape)
	row_sums_A = A.sum(axis = 1)
	denom = numpy.square(row_sums_A)
	row_sums_A_grad = df_dwk.sum(axis = 1)
	for i in range(df_dwk.shape[0]):
		for j in range(df_dwk.shape[1]):
			dQ_dwk_dense[i, j] = (1 - alpha) * (df_dwk[i, j] * row_sums_A[i] - A[i, j] * row_sums_A_grad[i]) / denom[i]

	return scipy.sparse.csr_matrix(dQ_dwk_dense)
