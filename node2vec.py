

def node2vec_walk(self, walk_length, start_node):
    G = self.G    
    alias_nodes = self.alias_nodes    
    alias_edges = self.alias_edges
    walk = [start_node]
    while len(walk) < walk_length:        
        cur = walk[-1]        
        cur_nbrs = list(G.neighbors(cur))        
        if len(cur_nbrs) > 0:            
            if len(walk) == 1:                
                walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])            
            else:                
                prev = walk[-2]                
                edge = (prev, cur)                
                next_node = cur_nbrs[alias_sample(alias_edges[edge][0],alias_edges[edge][1])]                
                walk.append(next_node)        
        else:            
            break
    return walk

def get_alias_edge(self, t, v):
    G = self.G    
    p = self.p    
    q = self.q
    unnormalized_probs = []    
    for x in G.neighbors(v):        
        weight = G[v][x].get('weight', 1.0)# w_vx        
        if x == t:# d_tx == 0            
            unnormalized_probs.append(weight/p)        
        elif G.has_edge(x, t):# d_tx == 1            
            unnormalized_probs.append(weight)        
        else:# d_tx == 2            
            unnormalized_probs.append(weight/q)    
    norm_const = sum(unnormalized_probs)    
    normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
    return create_alias_table(normalized_probs)

def preprocess_transition_probs(self):
    G = self.G
    alias_nodes = {}    
    for node in G.nodes():        
        unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]        
        norm_const = sum(unnormalized_probs)        
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]                 
        alias_nodes[node] = create_alias_table(normalized_probs)
    alias_edges = {}
    for edge in G.edges():        
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
    self.alias_nodes = alias_nodes    
    self.alias_edges = alias_edges
    return

# G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])
# model = Node2Vec(G,walk_length=10,num_walks=80,p=0.25,q=4,workers=1)
# model.train(window_size=5,iter=3)    
# embeddings = model.get_embeddings()
# evaluate_embeddings(embeddings)
# plot_embeddings(embeddings)