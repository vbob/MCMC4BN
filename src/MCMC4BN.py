import os
import pickle

import pandas as pd
import numpy as np
import pymc3 as pm

from pathlib import Path
from tqdm import tqdm
from itertools import combinations

from sklearn.model_selection import KFold

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore

from IPython.display import clear_output

class MCMC4BN:
    """
    Parameters:
        data (pd.DataFrame): Input data for the model
        dataset_name (string): Name of the dataset, used for saving results
        num_replicas (int): Number of resampled rdatasets
        num_samples (int): Number of samples per replica
        working_dir (string): Folder where results will be saved
    """
    def __init__(self, data, dataset_name, num_replicas=100, num_samples=100, working_dir='./output'):
        self.data = data
        self.dataset_name = dataset_name
        self.num_replicas = num_replicas
        self.num_samples = num_samples
        self.working_dir = working_dir + '/' + dataset_name + "_" + str(num_samples) + "/"
        
        self.networks = dict()
        self.possible_edges = [*combinations(data.columns, 2)]
        self.edge_counts = list()
        self.values = [-1, 0, 1]

        Path(self.working_dirfolder).mkdir(parents=True, exist_ok=True)    

    """
    Generate the replica datasets and networks

    Parameters:
        estimator (pgmpy.BaseEstimator): The search algorithm used
        score (pgmpy.BaseScore): The network evaluation score used
    """
    def generate_networks(self, estimator=HillClimbSearch, score=BDeuScore, seed=None, replace=True):
        # Generate and save the resampled datasets
        for i in tqdm(range(self.num_replicas)):
            df = self.data.sample(self.num_samples, replace=replace, random_state=seed*i)    
            df.to_csv(self.working_dir + "data_" + str(i) + ".csv", index=False)
        
        # Generate and save the networks for each resampled dataset
        for i, file in tqdm(enumerate([*filter(lambda x: 'data_' in x, os.listdir(self.working_dir))])):
            df = pd.read_csv(self.working_dir + file)
            self.networks[i] = list()

            scoring_method = score(data=df)    
            est = estimator(data=df)
            self[i].append(
                est.estimate(scoring_method=scoring_method, max_indegree=5, show_progress=False)
            )

    """
    Count the occurrence of edges between all variable pairs
    """
    def count_edges(self):
        if (len(self.networks.keys()) == 0):
            raise Exception("No network generated")

        self.edge_counts = [[] for i in range(len(self.networks))]

        for i in tqdm(range(self.num_replicas)):
            for j, network in enumerate(self.edge_counts[i]):
                self.edge_counts[i].append([])
                
                for edge in self.possible_edges:
                    # A --> B = self.values[0]
                    if (edge in network.edges):
                        self.edge_counts[i][j].append(self.values[0])

                    # A <-- B = self.values[2]
                    elif ((edge[1], edge[0]) in network.edges):
                        self.edge_counts[i][j].append(self.values[2])

                    # A -x- B = self.values[1]
                    else: 
                        self.edge_counts[i][j].append(self.values[1])

        self.edge_counts = self.edge_counts([np.array(x) for x in self.edge_counts])

    """
    Parameters:
        values: list of possible values for each edge (ex: [0, 1, 2] or [-1, 0, 1])
        observed_networks: list of networks observed
        edge_index: edged which will be counted
    """    
    def get_edge_frequency(self, edge_index):
        edge_frequency = list()
        
        for edge_list in self.networks:
            edges_obs = edge_list[:, edge_index]

            rightEdge = np.sum(edges_obs == self.values[0])
            noEdge = np.sum(edges_obs == self.values[1])
            leftEdge = np.sum(edges_obs == self.values[2])

            edge_frequency.append([rightEdge, noEdge, leftEdge])
    
        return edge_frequency

    """
    Parameters:
        edge_index (int): The desired edge, based on the position in self.possible_edges
        chains (int): Number of Markov chains
        Tune (int): Number of burn-in draws
        Draws (int): Number of effective draws
        nthreads (int): Number of parallell threads
    """
    def learn_edge(self, edge_index, chains=10, tune=10000, draws=10000, nthreads=10):
        k = len(self.values)          
        n = self.num_replicas
        total_count = len(self.possible_edges)
        edge = self.possible_edges[edge_index]

        observed_edges = self.get_edge_frequency(edge_index)

        with pm.Model() as model_dm_explicit:
            frac = pm.Dirichlet("frac", a=np.ones(k))
            conc = pm.Lognormal("conc", mu=1, sigma=1)
            p = pm.Dirichlet("p", a=frac * conc, shape=(n, k))

            counts = pm.Multinomial("counts", n=total_count, p=p, shape=(n, k), observed=observed_edges)
            trace_dm_explicit = pm.sample(chains=chains, tune=tune, draws=draws, step=pm.NUTS(), return_inferencedata=False, cores=nthreads)
            
            with open(self.working_dir + str(edge_index) + '_trace_' + edge[0] + "-" + edge[1] + '.pickle', 'wb') as handle:
                pickle.dump({
                    'model': model_dm_explicit, 
                    'frac':  frac, 
                    'conc': conc, 
                    'p': p, 
                    'counts': counts, 
                    'trace': trace_dm_explicit
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            return pm.summary(trace_dm_explicit)

    """
    Parameters:
        chains (int): Number of Markov chains
        Tune (int): Number of burn-in draws
        Draws (int): Number of effective draws
        nthreads (int): Number of parallell threads
    """    
    def run(self, chains=10, tune=10000, draws=10000, nthreads=10):
        for i, edge in enumerate(tqdm(self.possible_edges)):
            self.learn_edge(i, chains=chains, tune=tune, draws=draws, nthreads=nthreads)
            clear_output(wait=True)