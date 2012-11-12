import numpy

#Note: p_grad is a matrix with a column for each feature (so column k is dp/dw_k)
#Note; Q_grad is a list of sparse matrices
def update_p_grad(p, p_grad, Q, Q_grad, num_features, params):
	print(p.shape)
	print(p_grad.shape)
	print(Q.shape)
	print(Q_grad[0].shape)
	partial_gradient_update_epsilon = params["partial_gradient_update_epsilon"]
	for k in range(num_features):
		t = 0
		while True:
			print(t)
			p_grad_column_next = (1 - params["teleport_prob"]) * Q.T.dot(numpy.reshape(p_grad[:, k], (p_grad.shape[0], 1))) + (params["teleport_prob"] * numpy.sum(p_grad[:, k]) / Q.shape[0]) * numpy.ones((Q.shape[0], 1))  + Q_grad[k].T.dot(p)
			print(p_grad_column_next.shape)
			diff = numpy.amax(numpy.fabs(p_grad_column_next.flatten() - p_grad[:, k]))
			print(diff)
			p_grad[:, k] = p_grad_column_next.flatten()
			print(p_grad)
			if diff < partial_gradient_update_epsilon:
				break

			t += 1

	return p_grad
