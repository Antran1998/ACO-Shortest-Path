#!/usr/bin/env python

from aco import Map, AntColony
import numpy as np


# Hardcoded parameters
ANTS = 50
ITERATIONS = 50
MAP_PATH = 'map3.txt'  # Adjust path relative to maps directory usage in Map class
P = 0.5        # Evaporation rate
Q = 10.0       # Pheromone deposit amount
DISPLAY = 1  # Set to 1 to display the map and path, 0 to disable
ALPHA = 1.0  # Pheromone influence
BETA = 5.0   # Heuristic influence
INITIAL_PHEROMONE = 1.0
MAX_STEPS = None         # use default grid^2 safeguard


if __name__ == '__main__':
    # Load map
    map_obj = Map(MAP_PATH)

    print("--- Running with Destination Boost ---")
    # Create colony with destination boost (default)
    colony_boosted = AntColony(map_obj, ANTS, ITERATIONS, P, Q,
                               initial_pheromone=INITIAL_PHEROMONE,
                               alpha=ALPHA, beta=BETA,
                               max_steps=MAX_STEPS,
                               destination_pheromone_boost=True,
                               destination_boost_radius = 15,
                               boost_factor=4.0)
    # Compute path
    path_boosted = colony_boosted.calculate_path()
    print('Resulting best path length (Boosted):', len(path_boosted))
    print('Best path (Boosted):', path_boosted)
    if DISPLAY > 0:
        map_obj.represent_path(path_boosted)

    print("\n--- Running without Destination Boost ---")
    # Create colony without destination boost
    colony_no_boost = AntColony(map_obj, ANTS, ITERATIONS, P, Q,
                                initial_pheromone=INITIAL_PHEROMONE,
                                alpha=ALPHA, beta=BETA,
                                max_steps=MAX_STEPS,
                                destination_pheromone_boost=False)
    # Compute path
    path_no_boost = colony_no_boost.calculate_path()
    print('Resulting best path length (No Boost):', len(path_no_boost))
    print('Best path (No Boost):', path_no_boost)
    if DISPLAY > 0:
        map_obj.represent_path(path_no_boost)

