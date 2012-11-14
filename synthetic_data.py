import networkx as nx
import random



"""
  === load the scale-free graph G and its features ====
"""
def load_data(adjlist_file, feature_file):
    g = nx.read_adjlist(adjlist_file,  create_using=nx.Graph(), nodetype = int)
    features = {}
    f = open(feature_file)
    for line in f:
        n1,n2,f1,f2 = line.strip().split()
        features[(int(n1),int(n2))] = [float(f1),float(f2)]
    f.close()
    return g, features

"""
 ===  generate scale-free graph G ==== 
    input:
     - N : number of nodes for the network
     - alpha: probability that a new node selects a neighbor uniformly at random 
     - adjlist_file: name of the file that stores edges info of the network
         - format: <node 1> <node 2>
     - features_file: name of the file that stores feature info of the network
         - format: <node 1> <node 2> <feature 1> <feature 2>
      
    output:
    - graph: an undirected graph with N nodes 
    - features:  a dictionary of feature matrix (key = edge, value = list of 2 features)
"""
def create_synthetic_data(N, alpha, adjlist_file, features_file):
    #start with three nodes connected in a triad
    g = nx.complete_graph(3, create_using=nx.Graph())
    
    # add new nodes    
    while g.number_of_nodes() < N:
        new_nodeId = g.number_of_nodes()    
        old_nodes = g.nodes()
        
        i = 0
        while i < 3:
            if (random.uniform(0.0, 1.0) > alpha):
                """ pick an old nodes uniformly at random as neighbor"""        
                new_neighbor = random.choice(old_nodes)
            else:
                """ select neighbor with probability proportional to its current degree """            
                pool = []             
                for node in old_nodes:                
                    pool.extend([node]*g.degree(node))
                new_neighbor = random.choice(pool)            
                
            if not g.has_edge(new_nodeId, new_neighbor):
                g.add_edge(new_nodeId, new_neighbor)    
                i += 1
    
    # generate features
    edges = g.edges()
    features = {}
    mu = 0
    sigma = 1 
    for edge in edges:
        features[edge] = [random.gauss(mu, sigma), random.gauss(mu, sigma)]

                
    # save the graph and features        
    adjlist_out = open(adjlist_file, 'w+')
    feature_out = open(features_file, 'w+')
    
    edges = g.edges()
    for edge in edges:
        n1,n2 = edge
        adjlist_out.write('%i %i\n'%(n1, n2))
        
        f = features[edge]
        feature_out.write('%i %i %f %f\n'%(n1, n2, f[0], f[1] ))
    adjlist_out.close()
    feature_out.close()    

    return g,features

        

# ======= example =======================
if __name__ == '__main__':
    N = 10000
    alpha = 0.8
    adjlist_file = 'synthetic_data.ajdlist'
    features_file = 'synthetic_data.feature'
    
    # creating lots of graphs
    num_of_graphs = 100
    for i in xrange(num_of_graphs):        
        print 'generating ', i, '...'        
        g, f = create_synthetic_data(N, alpha, adjlist_file + str(i), features_file + str(i))
        print g.number_of_nodes(), g.number_of_edges()
    
    
    # test for loading data
    #g2, f2 = load_synthetic_data(adjlist_file, features_file)
    #print g2.number_of_nodes()
    #print g2.number_of_edges()
    
    print 'done!'    
