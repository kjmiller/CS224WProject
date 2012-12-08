import edge_computation
import cost_one_source
import grad_one_source
import numpy

def setup_params():
	params = {}
	params["pool_size"] = 8
	params["edge_strength_fun"] = edge_computation.logistic
	params["loss_fun"] = cost_one_source.sigmoid_h_loss
	params["h_grad_fun"] = grad_one_source.sigmoid_h_loss_grad
	params["partial_gradient_update_epsilon"] = 1e-12
	params["page_rank_epsilon"] = 1e-12
	params["backprop"] = 1
	params["backprop_epsilon"] = 1e-10
	params["edge_strength_grad_fun"] = edge_computation.logistic_grad
	params["maxiter"] = 1000
	params["alpha"] = 0.2
	params["teleport_prob"] = 0.15
	params["lambda"] = 1.0
	params["margin"] = 1.0
	params["K"] = 1000
	params["prop_training_gts"] = 0.2
	params["stationary_p"] = 1
	return params
