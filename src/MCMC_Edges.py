import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import scipy.stats
import seaborn as sns
import pandas as pd

from graphviz import Digraph
from itertools import combinations

def get_edge_frequency(values, observed_networks, edge_index):
    """
    values: list of possible values for each edge (ex: [0, 1, 2] or [-1, 0, 1])
    observed_networks: list of networks observed
    edge_index: edged which will be counted
    """
    
    observations = list()
    
    for edge_list in observed_networks:
        edges_obs = edge_list[:, edge_index]

        rightEdge = np.sum(edges_obs == values[0])
        noEdge = np.sum(edges_obs == values[1])
        leftEdge = np.sum(edges_obs == values[2])

        observations.append([rightEdge, noEdge, leftEdge])
    
    return observations



def model_edge(values, n_individuals, n_networks, edge_index, observed_networks):
    """
    values: list of possible values for each edge (ex: [0, 1, 2] or [-1, 0, 1])
    n_individuals: number of individuals which will be combined
    n_networks: number of edges for each individual
    edge_index: edge which will be modeled
    observed_networks: list of networks observed
    """
    
    observed_edges = get_edge_frequency(values, observed_networks, edge_index)
    
    k = len(values)          
    n = n_individuals        
    total_count = n_networks
    
    with pm.Model() as model_dm_explicit:
        frac = pm.Dirichlet("frac", a=np.ones(k))
        conc = pm.Lognormal("conc", mu=1, sigma=1)
        p = pm.Dirichlet("p", a=frac * conc, shape=(n, k))
        counts = pm.Multinomial("counts", n=total_count, p=p, shape=(n, k), observed=observed_edges)
        
    with model_dm_explicit:
        trace_dm_explicit = pm.sample(chains=4, return_inferencedata=True)
        
    return (model_dm_explicit, trace_dm_explicit)