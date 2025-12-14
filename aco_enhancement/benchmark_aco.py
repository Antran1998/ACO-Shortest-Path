import sys
import os
# Add parent directory to path to import aco module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from aco.map_class import Map
from ant_colony_enhancement import AntColony
from smooth_path_bspline import smooth_path_bspline

MAP_FILE = 'map4.txt'
RUNS = 30  # paper
NO_ANTS = 50       # Paper Table 1: Colony size = 50
EVAPORATION = 0.15 # Paper Table 1: Evaporation rate = 0.15
ITERATIONS = 100   # Paper: Convergence around 23-81 iterations
INIT_PHER = 0.0001 # Paper text: Initial concentration 1e-4

def run_single_config(name, map_obj, use_cone_pher=False, use_adaptive_proc=False, use_div_labor=False, use_deadlock=False): #Add use_deadlock argument for Improve 4
    print(f" Running: {name}")
    lengths = []
    times = []
    best_path = None
    best_length = float('inf')
    
    for r in range(RUNS):
        start_t = time.time()
        # Optimized parameters based on ACO theory:
        # - Moderate ant count (30 for 31x31 map) for efficiency
        # - Higher iterations (100) for better convergence
        # - Moderate evaporation (0.3) to balance exploration/exploitation
        # - Lower pheromone constant (5) to avoid premature convergence
        # - Higher beta (6) for strong heuristic guidance
        # - Moderate alpha (1.0) for balanced pheromone influence
        # - Higher xi (0.5) for stronger adaptive effect
            
        aco = AntColony(map_obj, 
                        no_ants=NO_ANTS,          # Updated to 50
                        iterations=ITERATIONS,  
                        evaporation_factor=EVAPORATION, # Updated to 0.15
                        pheromone_adding_constant=10.0,  
                        initial_pheromone=INIT_PHER,    # Updated to 0.0001
                        alpha=1.0, # Base alpha
                        beta=4.0,  # Adjusted beta for Euclidean distance logic
                        
                        # Feature Flags
                        use_cone_pheromone=use_cone_pher,
                        xi=0.5,  
                        use_adaptive_processing=use_adaptive_proc,
                        use_division_of_labor=use_div_labor,
                        use_deadlock_recovery=use_deadlock) # Improve 4 flag
        
        path = aco.calculate_path()
        end_t = time.time()
        
        if path:
            # Use Euclidean distance instead of node count (len)
            # This is crucial for scientific accuracy with diagonal movement
            path_len = aco.calculate_path_distance(path)
            
            lengths.append(path_len)
            times.append(end_t - start_t)
            
            # Track the best path
            if path_len < best_length:
                best_length = path_len
                best_path = path
            
    return {
        "name": name,
        "mean_len": np.mean(lengths) if lengths else float('inf'),
        "std_len": np.std(lengths) if lengths else 0,
        "min_len": np.min(lengths) if lengths else float('inf'),
        "mean_time": np.mean(times) if times else 0,
        "best_path": best_path
    }

def main():
    try:
        map_obj = Map(MAP_FILE)
    except FileNotFoundError:
        print(f"Error: Map file not found at maps/{MAP_FILE}")
        return
    
    # Start node + gold node
    start_node = map_obj.initial_node
    goal_node = map_obj.final_node
    print(f"Map Loaded: {MAP_FILE}")
    print(f"Start Node: {start_node} (Row, Col)")
    print(f"Goal Node : {goal_node} (Row, Col)")

    results = []

    # 1. test base ACO (no improvements)
    # Args: cone, adaptive, division, deadlock
    results.append(run_single_config("Base ACO", map_obj, False, False, False, False))

    # 2. test improve 1 (cone pheromone only - improve 1)
    results.append(run_single_config("Cone Pheromone", map_obj, True, False, False, False))

    # 3. test improve 2 (Adaptive pheromone - improve 2)
    results.append(run_single_config("Adaptive Processing", map_obj, False, True, False, False))

    # 4. test improve 3 (division of labor only - improve 3)
    results.append(run_single_config("Division of Labor", map_obj, False, False, True, False))
    
    # 5. test improve 4 (deadlock recovery only - improve 4)
    results.append(run_single_config("Deadlock Recovery", map_obj, False, False, False, True))

    # 6. test improve 6 (Proposed Method: All features combined)
    # Paper Fig 3
    results.append(run_single_config("Proposed Method", map_obj, True, True, True, True))

    # print benchmark results table
    print("\n" + "-"*85)
    print(f"{'Algorithm':<20} | {'Min Len':<10} | {'Mean Len':<15} | {'Time (s)':<10}")
    print("-" * 85)
    for res in results:
        print(f"{res['name']:<20} | {res['min_len']:<10.2f} | {res['mean_len']:<7.2f} ± {res['std_len']:<5.2f} | {res['mean_time']:<10.3f}")
    print("-"*85)

    # test improve 5 (smoothing)
    print("\n>>> Testing Improve 5 (Smoothing) on Best Path from Proposed Method...")
    
    # Get the best path from the Proposed Method configuration
    proposed_result = results[-1]  # Last result is Proposed Method
    raw_path = proposed_result["best_path"]
    
    if raw_path:
        # convert coords (row, col) to (x, y)
        path_y = [p[0] for p in raw_path]
        path_x = [p[1] for p in raw_path]
        path_xy = list(zip(path_x, path_y))
        
        # apply smoothing function
        smooth_path = smooth_path_bspline(path_xy)
        
        # Calculate length for smooth path for comparison
        smooth_len = 0
        for i in range(len(smooth_path)-1):
             smooth_len += np.sqrt((smooth_path[i+1][0]-smooth_path[i][0])**2 + 
                                   (smooth_path[i+1][1]-smooth_path[i][1])**2)
        
        print(f"Original Length: {proposed_result['min_len']:.2f}")
        print(f"Smoothed Length: {smooth_len:.2f}")
        
        # plot and save figure
        plt.figure(figsize=(10, 10))
        
        # plot map background
        plt.imshow(map_obj.occupancy_map, cmap='gray', origin='upper')
        
        # plot original path (red dashed)
        plt.plot(path_x, path_y, 'r--', linewidth=1.5, label='Proposed (Discrete)')
        
        # plot smoothed b-spline path (green solid)
        plt.plot(smooth_path[:, 0], smooth_path[:, 1], 'g-', linewidth=2.5, label='B-Spline (Smoothed)')
        
        # plot start/goal points
        plt.plot(start_node[1], start_node[0], 'bo', markersize=10, label='Start')
        plt.plot(goal_node[1], goal_node[0], 'r*', markersize=15, label='Goal')
        
        plt.legend()
        plt.title(f"Improve 5: Path Smoothing (Len: {proposed_result['min_len']:.2f} -> {smooth_len:.2f})")
        plt.xlabel("X (Column)")
        plt.ylabel("Y (Row)")

        # save to file
        plt.savefig('smoothing_comparison.png', dpi=300)
        print("\n[INFO] Comparison chart saved to file: smoothing_comparison.png")
        
        plt.show()
    else:
        print("[WARN] No path found for Proposed Method. Cannot apply smoothing.")

if __name__ == '__main__':
    main()