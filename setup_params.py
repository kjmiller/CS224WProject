import edge_computation
import cost_one_source
import grad_one_source

def setup_params():
	params = {}
	params["pool_size"] = 0
	params["edge_strength_fun"] = edge_computation.logistic
	params["loss_fun"] = cost_one_source.sigmoid_h_loss
	params["h_grad_fun"] = grad_one_source.sigmoid_h_loss_grad
	params["partial_gradient_update_epsilon"] = 1e-12
	params["page_rank_epsilon"] = 1e-12
	params["edge_strength_grad_fun"] = edge_computation.logistic_grad
	params["maxiter"] = 1000
	params["alpha"] = 0.2
	params["teleport_prob"] = 0.15
	params["lambda"] = 1.0
	params["margin"] = 100.0 #1.0
	params["K"] = 5000
	return params
