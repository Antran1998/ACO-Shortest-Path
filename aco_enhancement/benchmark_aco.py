import sys
import os
# Add parent directory to path to import aco module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import time
import matplotlib.pyplot as plt
from aco.map_class import Map
from smooth_path_bspline import smooth_path_bspline


from importlib import import_module

# Select appropriate AntColony class based on method 
def get_ant_colony_instance(name, map_obj, O1, O2, O3, O4, O5, O6):
    if name.lower().startswith("basic aco"):
        AntColonyClass = import_module("aco.ant_colony").AntColony
        return AntColonyClass(map_obj, 
                             no_ants=NO_ANTS,
                             iterations=ITERATIONS,  
                             evaporation_factor=EVAPORATION, 
                             pheromone_adding_constant=10.0,  
                             alpha=1.0, 
                             beta=4.0,  
                             xi=0.5,
                             initial_pheromone=INIT_PHER)
    else:
        AntColonyClass = import_module("aco_enhancement.ant_colony_enhancement").AntColony
        return AntColonyClass(map_obj, 
                             no_ants=NO_ANTS,
                             iterations=ITERATIONS,  
                             evaporation_factor=EVAPORATION, 
                             pheromone_adding_constant=10.0,  
                             initial_pheromone=INIT_PHER,    
                             alpha=1.0, 
                             beta=4.0,  
                             xi=0.5,  
                             use_cone_pheromone=O1,
                             use_adaptive_processing=O2,
                             use_division_of_labor=O3,
                             use_backtracking=O4)

MAP_FILE = 'map1.txt'
RUNS = 30  # paper
NO_ANTS = 50       # Paper Table 1: Colony size = 50
EVAPORATION = 0.15 # Paper Table 1: Evaporation rate = 0.15
ITERATIONS = 100   # Paper: Convergence around 23-81 iterations
INIT_PHER = 0.0001 # Paper text: Initial concentration 1e-4

METHODS = [
    # (method_id, method_name, (O1, O2, O3, O4, O5, O6), description)
    (1, "Basic ACO",                (False, False, False, False, False, False), "No improvements. All options off. Standard ACO."),
    (2, "Cone Pheromone (O1)",      (True,  False, False, False, False, False), "Cone pheromone initialization (O1) only."),
    (3, "Adaptive Heuristic (O2)",  (False, True,  False, False, False, False), "Adaptive heuristic factor (O2) only."),
    (4, "Division of Labor (O3)",   (False, False, True,  False, False, False), "Division of labor (O3) only."),
    (5, "Backtracking (O4)",        (False, False, False, True,  False, False), "Deadlock/backtracking (O4) only."),
    (6, "Path Smoothing (O5)",      (False, False, False, False, True,  False), "Path smoothing (O5) only. Smoothing applied after basic ACO."),
    (7, "New Algorithm Flow (O6)",  (False, False, False, False, False, True),  "Algorithm step optimization (O6) only."),
    (8, "Proposed Method (Full)",   (True,  True,  True,  True,  True,  True),  "All improvements enabled (O1-O6)."),
]

def gen_tag(method_id):
    return '' if method_id == 1 else f"m{method_id}-"

def run_single_config(name, map_obj, O1, O2, O3, O4, O5, O6):
    print(f"\nRunning: {name}")
    lengths = []
    times = []
    best_path = None
    best_length = float('inf')
    for r in range(RUNS):
        # Progress indicator (overwrite line)
        print(f"  Progress: {r+1}/{RUNS}...", end='\r', flush=True)
        start_t = time.time()
        aco = get_ant_colony_instance(name, map_obj, O1, O2, O3, O4, O5, O6)
        path = aco.calculate_path()
        end_t = time.time()
        if path:
            path_len = aco.calculate_path_distance(path)
            lengths.append(path_len)
            times.append(end_t - start_t)
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

def load_map(map_file):
    try:
        map_obj = Map(map_file)
    except FileNotFoundError:
        print(f"Error: Map file not found at maps/{map_file}")
        return None
    return map_obj

def prompt_method_selection(methods):
    print("\nSelect method to benchmark:")
    for m in methods:
        print(f"  {m[0]}: {m[1]} - {m[3]}")
    while True:
        try:
            mode = int(input(f"Enter the number corresponding to the method you want to run (1-{len(methods)}): "))
            if mode in range(1, len(methods)+1):
                return mode
            else:
                print(f"Invalid input. Please enter a number from 1 to {len(methods)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number from 1 to {len(methods)}.")

def print_results_table(results):
    print("\n" + "-"*70)
    print(f"{'Algorithm':<20} | {'Mean Len':<15} | {'Time (s)':<10}")
    print("-" * 70)
    for res in results:
        print(f"{res['name']:<20} | {res['mean_len']:<7.2f} ± {res['std_len']:<5.2f} | {res['mean_time']:<10.3f}")
    print("-"*70)

def save_stats_table(result_dir, map_base, method_file, tag, method_id, method_name, method_desc, results):
    stat_filename = f"stat-{map_base}-{tag}{method_file}.txt"
    stat_path = os.path.join(result_dir, stat_filename)
    with open(stat_path, 'w', encoding='utf-8') as f:
        f.write(f"Method {method_id}: {method_name}\n{method_desc}\n\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Algorithm':<20} | {'Mean Len':<15} | {'Time (s)':<10}\n")
        f.write("-" * 70 + "\n")
        for res in results:
            f.write(f"{res['name']:<20} | {res['mean_len']:<7.2f} ± {res['std_len']:<5.2f} | {res['mean_time']:<10.3f}\n")
        f.write("-"*70 + "\n")

def recalc_stats_with_smoothing(results):
    """
    Recalculate mean_len, std_len, mean_time based on the smoothed (B-Spline) path.
    mean_time = original mean_time (finding best path) + smoothing time.
    """
    from smooth_path_bspline import smooth_path_bspline
    import numpy as np
    import time
    for res in results:
        best_path = res.get('best_path')
        if best_path is not None:
            # Convert (row, col) to (x, y)
            path_y = [p[0] for p in best_path]
            path_x = [p[1] for p in best_path]
            path_xy = list(zip(path_x, path_y))
            # Measure smoothing time
            t0 = time.time()
            smooth_path = smooth_path_bspline(path_xy)
            t1 = time.time()
            smooth_time = t1 - t0
            # Calculate smoothed path length
            smooth_len = 0
            for i in range(len(smooth_path)-1):
                smooth_len += np.sqrt((smooth_path[i+1][0]-smooth_path[i][0])**2 + (smooth_path[i+1][1]-smooth_path[i][1])**2)
            res['mean_len'] = smooth_len
            res['std_len'] = 0
            res['mean_time'] = res.get('mean_time', 0) + smooth_time
    return results

def save_discrete_path_plot(result_dir, map_base, method_file, tag, method_name, min_len, best_path, map_obj, start_node, goal_node):
    path_y = [p[0] for p in best_path]
    path_x = [p[1] for p in best_path]
    plt.figure(figsize=(10, 10))
    plt.imshow(map_obj.occupancy_map, cmap='gray', origin='upper')
    plt.plot(path_x, path_y, 'r--', linewidth=2, label=method_name+' (Discrete)')
    plt.plot(start_node[1], start_node[0], 'bo', markersize=10, label='Start')
    plt.plot(goal_node[1], goal_node[0], 'r*', markersize=15, label='Goal')
    plt.legend()
    plt.title(f"{method_name} - Discrete Path (Len: {min_len:.2f})")
    img_filename = f"visu-{map_base}-{tag}{method_file}.png"
    img_path = os.path.join(result_dir, img_filename)
    plt.savefig(img_path, dpi=300)
    plt.close()

def save_smoothed_path_plot(result_dir, map_base, method_file, tag, method_name, min_len, best_path, map_obj, start_node, goal_node):
    print(f"\n>>> Testing Path Smoothing on Best Path from {method_name}...")
    if not best_path:
        print(f"[WARN] No path found for smoothing. raw_path={best_path}")
        return
    try:
        path_y = [p[0] for p in best_path]
        path_x = [p[1] for p in best_path]
        path_xy = list(zip(path_x, path_y))
        smooth_path = smooth_path_bspline(path_xy)
        smooth_len = 0
        for i in range(len(smooth_path)-1):
            smooth_len += np.sqrt((smooth_path[i+1][0]-smooth_path[i][0])**2 + \
                                  (smooth_path[i+1][1]-smooth_path[i][1])**2)
        print(f"Original Length: {min_len:.2f}")
        print(f"Smoothed Length: {smooth_len:.2f}")
        plt.figure(figsize=(10, 10))
        plt.imshow(map_obj.occupancy_map, cmap='gray', origin='upper')
        plt.plot(path_x, path_y, 'r--', linewidth=1.5, label=method_name+' (Discrete)')
        plt.plot(smooth_path[:, 0], smooth_path[:, 1], 'g-', linewidth=2.5, label='B-Spline (Smoothed)')
        plt.plot(start_node[1], start_node[0], 'bo', markersize=10, label='Start')
        plt.plot(goal_node[1], goal_node[0], 'r*', markersize=15, label='Goal')
        plt.legend()
        plt.title(f"{method_name} - Path Smoothing (Len: {min_len:.2f} -> {smooth_len:.2f})")
        img_filename = f"visu-{map_base}-{tag}{method_file}.png"
        img_path = os.path.join(result_dir, img_filename)
        plt.savefig(img_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Exception during PNG save: {e}")

def initialize_map():
    map_obj = load_map(MAP_FILE)
    if map_obj is None:
        return None
    map_shape = map_obj.in_map.shape if hasattr(map_obj, 'in_map') else (None, None)
    print(f"Loaded map: {MAP_FILE} | Size: {map_shape[0]}x{map_shape[1]}")
    return map_obj

def get_start_and_goal_nodes(map_obj):
    return map_obj.initial_node, map_obj.final_node

def select_method(methods):
    mode = prompt_method_selection(methods)
    method_id, method_name, flags, method_desc = methods[mode-1]
    return method_id, method_name, flags, method_desc

def execute_selected_method(method_name, map_obj, flags):
    O1, O2, O3, O4, O5, O6 = flags
    return [run_single_config(method_name, map_obj, O1, O2, O3, O4, O5, O6)]

def output_results(map_file, method_id, method_name, method_desc, results, map_obj, start_node, goal_node, flags):
    print_results_table(results)
    result_dir = os.path.join(os.path.dirname(__file__), '..', 'result')
    os.makedirs(result_dir, exist_ok=True)
    map_base = os.path.splitext(os.path.basename(map_file))[0]
    method_file = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').lower()
    tag = gen_tag(method_id)
    save_stats_table(result_dir, map_base, method_file, tag, method_id, method_name, method_desc, results)
    best_path = results[-1]["best_path"]
    O5 = flags[4]
    if best_path and not O5:
        save_discrete_path_plot(result_dir, map_base, method_file, tag, method_name, results[-1]['min_len'], best_path, map_obj, start_node, goal_node)
    elif not best_path:
        print("[WARN] No best path found to visualize for this method.")
    if O5:
        save_smoothed_path_plot(result_dir, map_base, method_file, tag, method_name, results[-1]['min_len'], best_path, map_obj, start_node, goal_node)

def main():
    map_obj = initialize_map()
    if map_obj is None:
        return
    start_node, goal_node = get_start_and_goal_nodes(map_obj)
    method_id, method_name, flags, method_desc = select_method(METHODS)
    results = execute_selected_method(method_name, map_obj, flags)
    output_results(MAP_FILE, method_id, method_name, method_desc, results, map_obj, start_node, goal_node, flags)

if __name__ == '__main__':
    main()

