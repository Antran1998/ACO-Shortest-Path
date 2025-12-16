#!/usr/bin/env python

from aco import Map
from aco import AntColony
import numpy as np
import sys
import argparse


if __name__ == '__main__':

    # === Senerio parameters ===
    no_ants = 10
    iterations = 50
    map_path = 'map3.txt'
    display = True

    # =============== Pheromone information ================
    #  tau^alpha <- tau_ij(t) = (1-rho) * tau_ij(t-1) + Q/L 
    #=======================================================

    rho = 0.15
    Q = 10  
    alpha = 1.0 

    # ================== Heuristic information =============
    #  eta^beta <- eta_ij = 1 / d_ij 
    #=======================================================

    beta = 2.0
    
    # ============ Adaptive heuristic factors ==============
    #  alpha' = alpha + xi*Integral{[0 to n/N] t dt}    
    #  beta' = beta + xi*Integral{[0 to n/N] t dt}
    #=======================================================
    
    xi = 0.01

    # Get the map and run
    map_obj = Map(map_path)

    Colony = AntColony(map_obj, 
                       no_ants, 
                       iterations, 
                       rho, 
                       Q, 
                       alpha, 
                       beta,
                       xi)
    
    path = Colony.calculate_path()
    print(path)
    if display > 0:
        map_obj.represent_path(path)

