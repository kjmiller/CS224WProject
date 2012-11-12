import numpy

def update_p(p, Q, params):
	page_rank_epsilon = params["page_rank_epsilon"]
	teleport_prob = params["teleport_prob"]
	while True:
		p_next = (1 - teleport_prob) * Q.T.dot(p) + teleport_prob * numpy.ones(p.shape) / (1.0 * Q.shape[0])
		diff = numpy.amax(numpy.fabs(p_next - p))
		p = p_next
		if diff < page_rank_epsilon:
			break

	return p
