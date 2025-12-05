import sys
import os
# Add parent directory to path to import aco module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
from aco.map_class import Map
from ant_colony_improve1_3_5 import AntColony
from smooth_path_bspline import smooth_path_bspline

MAP_FILE = 'map3.txt'
RUNS = 100  # run 5 times to get average metrics

def run_single_config(name, use_imp1, use_imp3, map_obj):
    print(f" Running: {name}")
    lengths = []
    times = []
    
    for r in range(RUNS):
        start_t = time.time()
        # optimized parameters for 31x31 map
        # increased ants for better exploration, optimized evaporation
        aco = AntColony(map_obj, 60, 40, 0.3, 12.0, 
                        initial_pheromone=1.5,
                        alpha=1.0,
                        beta=2.5,
                        use_cone_pheromone=use_imp1, 
                        use_division_of_labor=use_imp3,
                        destination_boost_radius=20,
                        boost_factor=6.0)
        
        path = aco.calculate_path()
        end_t = time.time()
        
        if path:
            lengths.append(len(path))
            times.append(end_t - start_t)
            
    return {
        "name": name,
        "mean_len": np.mean(lengths) if lengths else float('inf'),
        "min_len": np.min(lengths) if lengths else float('inf'),
        "mean_time": np.mean(times) if times else 0
    }

def main():
    try:
        map_obj = Map(MAP_FILE)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file maps/{MAP_FILE}")
        return
    
    # Start node + gold node
    start_node = map_obj.initial_node
    goal_node = map_obj.final_node
    print(f"Map Loaded: {MAP_FILE}")
    print(f"Start Node: {start_node} (Row, Col)")
    print(f"Goal Node : {goal_node} (Row, Col)")

    results = []

    # 1. test base ACO (no improvements)
    results.append(run_single_config("Base ACO", False, False, map_obj))

    # 2. test improve 1 (cone pheromone only - improve 1)
    results.append(run_single_config("ACO + Imp1 (Cone)", True, False, map_obj))

    # 3. test improve 3 (division of labor only - improve 3)
    results.append(run_single_config("ACO + Imp3 (DivLab)", False, True, map_obj))

    # 4. test combined features
    results.append(run_single_config("ACO + Imp1 + Imp3", True, True, map_obj))

    # print benchmark results table
    print("\n" + "-"*60)
    print(f"{'Algorithm':<20} | {'Min Len':<10} | {'Mean Len':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['name']:<20} | {res['min_len']:<10.1f} | {res['mean_len']:<10.1f} | {res['mean_time']:<10.3f}")
    print("-"*60)

    # test improve 5 (smoothing) separately
    print("\n>>> Testing Improve 5 (Smoothing) on Best Path from Imp3...")
    
    # rerun once using best algorithm (imp1+imp3) with optimized params
    aco_best = AntColony(map_obj, 60, 40, 0.3, 12.0,
                         initial_pheromone=1.5,
                         alpha=1.0,
                         beta=2.5, 
                         use_cone_pheromone=True, 
                         use_division_of_labor=True,
                         destination_boost_radius=20,
                         boost_factor=6.0)
    raw_path = aco_best.calculate_path()
    
    if raw_path:
        # convert coords (row, col) to (x, y)
        path_y = [p[0] for p in raw_path]
        path_x = [p[1] for p in raw_path]
        path_xy = list(zip(path_x, path_y))
        
        # apply smoothing function
        smooth_path = smooth_path_bspline(path_xy, map_obj)
        
        print(f"Original Nodes: {len(raw_path)}")
        print(f"Smoothed Nodes: {len(smooth_path)}")
        
        # plot and save figure
        plt.figure(figsize=(10, 10))
        
        # plot map background
        plt.imshow(map_obj.occupancy_map, cmap='gray', origin='upper')
        
        # plot original path (red dashed)
        plt.plot(path_x, path_y, 'r--', linewidth=1.5, label='Original ACO (Discrete)')
        
        # plot smoothed b-spline path (green solid)
        # note: smooth_path is a numpy array
        plt.plot(smooth_path[:, 0], smooth_path[:, 1], 'g-', linewidth=2.5, label='B-Spline (Smoothed)')
        
         # plot start/goal points
        plt.plot(start_node[1], start_node[0], 'bo', markersize=10, label='Start')
        plt.plot(goal_node[1], goal_node[0], 'r*', markersize=15, label='Goal')
        
        plt.legend()
        plt.title(f"Improve 5: Path Smoothing (Nodes: {len(raw_path)} -> {len(smooth_path)})")
        plt.xlabel("X (Column)")
        plt.ylabel("Y (Row)")

        # save to file
        plt.savefig('smoothing_comparison.png', dpi=300)
        print("\n[INFO] Comparison chart saved to file: smoothing_comparison.png")
        
        plt.show()

if __name__ == '__main__':
    main()