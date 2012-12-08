import os
import os.path
import sys
import setup_params
import problem_setup_for_coauthorship
import problem_setup
import numpy.random
import numpy
import networkx as nx
import predict_one_source
import copy
import supervised_random_walk
import compute_cost
import compute_grad
import grad_one_source
import multiprocessing
import time


def load_coauthorship_graph(infile, node_mapper):    
    g = nx.Graph()    
    f = open(infile)         
                
    for line in f:
        if line[0] == '#':
            continue
        
        segs = line.strip().split('\t')
        
        node1, node2, first_time, last_time, num_coautho, paper_ids, all_times =  segs                               
        g.add_edge(node_mapper[long(node1)],node_mapper[long(node2)])
    f.close()
    return g


if __name__ == '__main__':
    
    # parameter for coauthorship
    year = 1980
#    home_dir = "D:/Dropbox/CS224W_Project/coauthorship_data/%s"%year
    home_dir = "../%s"%year 
    active_author_file_name = home_dir + "/dblp_coauthor.active_authors.%s"%year
    adjlist_file = home_dir + "/dblp_coauthor.adjlist_%s"%year
    node_mapper_file = home_dir +"/dblp_coauthor.node_mapper_%s"%year
    feature_file_prefix =home_dir + "/dblp_coauthor.features.%s"%year
     
    num_features = 6
    
    
    # start time for measuring the process time
    start = time.clock()
        
    # input arguments    
    numpy.random.seed(int(sys.argv[1]))
    sigma_squared = float(sys.argv[2])
    
    # load parameters for the random walk or backprop    
    params = setup_params.setup_params()
    pool = multiprocessing.Pool(8)
        
    # load node mapper
    node_mapper = {} # map from orig id to consecutive id
    invert_node_mapper = {} # map from consecutive id to orig id 
    f = open(node_mapper_file)
    for line in f:
        orig_id, new_id = line.split()
        node_mapper[long(orig_id)] = long(new_id)
        invert_node_mapper[long(new_id)] = long(orig_id)
        
    # load graph     
    orig_G = load_coauthorship_graph(adjlist_file, node_mapper)        
    num_nodes = orig_G.number_of_nodes()
    
    # read in the active authors
    active_authors_id_map = {} # map from the author id in adjlist to a continuous id 
    active_authors = {} # same as the active author list
    active_author_file = open(active_author_file_name, "r")
    for line in active_author_file:
        author, pos_str, neg_str, tu = line.split(';')
        author = node_mapper[long(author)]
            
        pos = pos_str.split(',')
        neg = neg_str.split(',')
        if pos[0] == '' or neg[0] == '':                    
            continue # has 0 positive labels or 0 negative labels for this author, skip
                
        active_authors[author] = {'training_positives': [node_mapper[long(p)] for p in pos], 
                                  'training_negatives': [node_mapper[long(n)] for n in neg]}                                                                    
                
    print "number of valid active authors:", len(active_authors)
    
    
    
    # generate training data list
    print "generating training data list"
    i = 0        
    training_data_list =[]    
    D_sum = 0.0
    C_sum = 0.0
    for author_id in active_authors.keys():
        feature_file_name = "%s.%s"%(feature_file_prefix,invert_node_mapper[author_id])
                
        if not os.path.exists(feature_file_name):
            #print "no such feature file:%s"%feature_file_name
            del active_authors[author_id]              
            continue
                        
        
        (edge_ij, feature_stack, G) = problem_setup_for_coauthorship.get_edge_ij_and_feature_stack(orig_G, feature_file_name, num_features, node_mapper)        
        source = author_id
        candidates = list(active_authors[author_id]["training_positives"])
        candidates.extend(active_authors[author_id]["training_negatives"])
        
        training_data_list.append({})                        
        training_data_list[i]["training_positives"] = active_authors[author_id]["training_positives"]
        training_data_list[i]["training_negatives"] = active_authors[author_id]["training_negatives"]                
        training_data_list[i]["num_features"] = num_features
        training_data_list[i]["source"] = source
        training_data_list[i]["candidates"] = candidates
        training_data_list[i]["edge_ij"] = edge_ij
        training_data_list[i]["feature_stack"] = feature_stack
        training_data_list[i]["num_nodes"] = num_nodes
                
        D_sum += len(active_authors[author_id]["training_positives"])
        C_sum += len(candidates)
    
        i += 1
        
    print "dataset statistics:"
    D = D_sum/len(training_data_list)
    C = C_sum/len(training_data_list)
    print "N:", orig_G.number_of_nodes()
    print "E:", orig_G.number_of_edges()
    print "S:", len(training_data_list)
    print "D:", D
    print "C:", C
    print "D/C:", D/C
    
        
    for training_data in training_data_list:
        rpos = numpy.random.rand(len(training_data["training_positives"]))
        rneg = numpy.random.rand(len(training_data["training_negatives"]))
        training_data["positives"] = []
        for i in range(len(rpos)):
            if rpos[i] <= params["prop_training_gts"]:
                training_data["positives"].append(training_data["training_positives"][i])
    
        training_data["negatives"] = []
        for i in range(len(rneg)):
            if rneg[i] <= params["prop_training_gts"]:
                training_data["negatives"].append(training_data["training_negatives"][i])
                
        training_data["diff_generating_mat"] = grad_one_source.build_diff_generating_mat(training_data["positives"], training_data["negatives"], training_data["num_nodes"])
    
    
    p_warm_start_list = [numpy.ones((num_nodes, 1)) / (1.0 * num_nodes)] * len(training_data_list)
    p_grad_warm_start_list = [numpy.zeros((num_nodes, training_data_list[0]["num_features"]))] * len(training_data_list)
    
    
    print "start training..."
    w0 = numpy.random.randn(6, 1) #numpy.array([[1.0], [-1.0]])
    print "w0=",w0
    
    w_opt = supervised_random_walk.train(training_data_list, p_warm_start_list, p_grad_warm_start_list, params, compute_cost.compute_cost, compute_grad.compute_grad, w0, pool = pool)
    
    numpy.save("model_coauthorship_backprop_mini_%f.npy"%(sigma_squared), w_opt)
    
    print "Total time: %.6f sec "%((time.clock() - start))

