from networkx.classes.function import nodes
import numpy as np
import os
import networkx as nx
import random
from tqdm import tqdm
from gensim.models import Word2Vec

class DeepWalk():
    ## deepwalk model 
    def __init__(self, node_num: int, edge_txt: str, undirected = False) -> None:
        ## graph init
        if undirected:
            ## Undirected graph
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        self.G.add_nodes_from(list(range(node_num)))
        
        ## read edges
        edges = read_edge(edge_txt=edge_txt)
        self.G.add_edges_from(edges)
        
        ## get graph information
        self.adjacency = np.array(nx.adjacency_matrix(self.G).todense())
        self.G_neighbor = {}
        for i in range(self.adjacency.shape[0]):
            
            self.G_neighbor[i] = []
            
            for j in range(self.adjacency.shape[0]):
                if i == j:
                    ## without self loop
                    continue
                
                if self.adjacency[i, j] > 0.01:
                    self.G_neighbor[i].append(j)
                    
def read_edge(edge_txt):
    ## implemented by customer according to edges information format in the txt file.
    edges = np.loadtxt(edge_txt, dtype=np.int16)
    edges = [(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])]
    
    return edges 

def random_walk(self, path_len, alpha=0, rand_iter=random.Random(9931), start=None):
    """ Returns a random walk with fixed length.
        path_len: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self.G_neighbor
    if start:
        rand_path = [start]
    else:
        # Sampling is uniform w.r.t V, and not w.r.t E
        rand_path = [rand_iter.choice(list(G.keys()))]

    while len(rand_path) < path_len:
        current_pos = rand_path[-1]
        if len(G[current_pos]) > 0:
            # current node have neighbors.
            if rand_iter.random() >= alpha:
                rand_path.append(rand_iter.choice(G[current_pos]))
            else:
                # restart from the start point.
                rand_path.append(rand_path[0])
        else:
            # stop when there is no other neighbor to go.
            rand_path.append(rand_iter.choice(list(G.keys())))
            break
        
    return [str(node) for node in rand_path]

def build_total_corpus(
        self,
        num_paths: "how many path start from every node",
        path_length,
        alpha=0,
        rand_iter=random.Random(9931),
    ):
        ## build corpus
        print("Start randomwalk.")
        total_walks = []
        G = self.G_neighbor

        nodes = list(G.keys())

        for cnt in tqdm(range(num_paths)):
            print(len(nodes))
            rand_iter.shuffle(nodes)
            for node in nodes:
                # for every node, create a random walk
                total_walks.append(
                    self.random_walk(
                        path_length, rand_iter=rand_iter, alpha=alpha, start=node
                    )
                )

        return total_walks

def train_deepwalk(self, total_walks, embedding_dim=64, window_size=3, output=""):
    print("Training deepwalk.")
    model = Word2Vec(
        total_walks,
        size=embedding_dim,
        window=window_size,
        min_count=0,
        sg=1,
        hs=1,
        workers=8,
    )

    model.wv.save_word2vec_format(output)
    
# node_num = 100
# edge_test = 'test_edges.txt'
# walker = DeepWalk(node_num=node_num , edge_txt=edge_test)
# deepwalk_corpus = walker.build_total_corpus(2, 10)
# walker.train_deepwalk(deepwalk_corpus, embedding_dim = 64, window_size = 3, output = 'test.embeddings')    