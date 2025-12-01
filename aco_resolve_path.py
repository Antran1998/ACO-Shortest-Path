#!/usr/bin/env python

from aco import Map, AntColony
import numpy as np


# Hardcoded parameters
ANTS = 50
ITERATIONS = 100
MAP_PATH = 'map3.txt'  # Adjust path relative to maps directory usage in Map class
P = 0.2        # Adjusted evaporation for more memory of past paths
Q = 10.0      # Pheromone addition amount
DISPLAY = 1  # Set to 1 to display the map and path, 0 to disable
ALPHA = 1.0  # Pheromone influence
BETA = 2.0  # Heuristic influence
INITIAL_PHEROMONE = 1.0
MAX_STEPS = None         # use default grid^2 safeguard


if __name__ == '__main__':
    # Load map
    map_obj = Map(MAP_PATH)
    # Create colony
    colony = AntColony(map_obj, ANTS, ITERATIONS, P, Q,
                       initial_pheromone=INITIAL_PHEROMONE,
                       alpha=ALPHA, beta=BETA,
                       max_steps=MAX_STEPS)
    # Compute path
    path = colony.calculate_path()
    print('Resulting best path length:', len(path))
    print('Best path:', path)
    if DISPLAY > 0:
        map_obj.represent_path(path)

