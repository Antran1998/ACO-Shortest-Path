import numpy as np
import matplotlib.pyplot as plt
import math

# Import modules
from map_class import Map
from ant_colony import AntColony
from improve5 import smooth_path_bspline
from dwa import DWAConfig, RobotState, improved_dwa_control, motion_model

MAP_FILE = 'map3.txt'
ANTS = 50
ITERATIONS = 30
DWA_MAX_STEPS = 1000 

def main():
    # 1. global planning
    print("1. Global Planning...")
    map_obj = Map(MAP_FILE)
    
    # enable both cone pheromone and division of labor for optimal results
    aco = AntColony(map_obj, ANTS, ITERATIONS, 0.5, 10.0, 
                    alpha=1.0, beta=2.0, 
                    use_cone_pheromone=True,
                    use_division_of_labor=True)
    raw_path = aco.calculate_path()
    
    if not raw_path:
        print("No path found!")
        return

    # 2. improve 5
    print("2. Improve 5...")
    # convert path nodes (row, col) -> (y, x)
    path_y = [p[0] for p in raw_path] # row maps to y
    path_x = [p[1] for p in raw_path] # col maps to x
    path_xy = list(zip(path_x, path_y))
    
    global_path_smooth = smooth_path_bspline(path_xy, map_obj)

    # 3. obstacle setup
    print("3. Obstacle Setup...")
    obs_indices = np.argwhere(map_obj.occupancy_map == 0)
    
    # create boundary walls to keep robot inside
    height, width = map_obj.occupancy_map.shape
    border_obs = []
    for r in range(-1, height + 1):
        border_obs.append([-1, r])      
        border_obs.append([width, r])   
    for c in range(-1, width + 1):
        border_obs.append([c, -1])      
        border_obs.append([c, height])  
    
    if len(obs_indices) > 0:
        # map obstacles: flip coordinates (row, col) to (x, y)
        map_obs = np.column_stack((obs_indices[:, 1], obs_indices[:, 0]))
        border_arr = np.array(border_obs)
        obstacles = np.vstack((map_obs, border_arr))
    else:
        obstacles = np.array(border_obs)

    # 4. dwa simulation
    print("4. DWA Simulation...")
    
    start_node = global_path_smooth[0]
    goal_node = global_path_smooth[-1]
    
    look_ahead_idx = min(5, len(global_path_smooth)-1)
    p1 = global_path_smooth[0]
    p2 = global_path_smooth[look_ahead_idx]
    initial_theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    
    print(f"Initial Theta calibrated to: {initial_theta:.2f} rad")

    # init robot with calculated angle
    robot = RobotState(x=start_node[0], y=start_node[1], theta=initial_theta, v=0.0, w=0.0)
    
    config = DWAConfig()
    config.min_speed = 0.0    # prevent backward motion
    config.max_speed = 1.0    # max speed
    config.max_yaw_rate = 1.5 # allow faster rotation to escape tight spots
    config.radius = 0.3       # helps robot fit through narrow gaps (grid size=1)
    
    # increase path weight to force tracking the planned path
    config.w_path_dir = 1.0   # prioritize following global plan
    config.w_target_heading = 0.2
    config.w_dist = 0.2       # slightly lower obstacle avoidance to allow movement near walls

    history_x = [robot.x]
    history_y = [robot.y]
    
    reached = False
    for i in range(DWA_MAX_STEPS):
        # find nearest point on smoothed path
        dists = np.linalg.norm(global_path_smooth - np.array([robot.x, robot.y]), axis=1)
        current_idx = np.argmin(dists)
        
        # look ahead window
        # if near goal, target goal directly
        look_ahead = 8 
        target_idx = min(current_idx + look_ahead, len(global_path_smooth) - 1)
        local_goal = global_path_smooth[target_idx]
        
        # run dwa
        u, _ = improved_dwa_control(robot, config, local_goal, obstacles, global_path_smooth)
        robot = motion_model(robot, u[0], u[1], config.dt)
        
        history_x.append(robot.x)
        history_y.append(robot.y)
        
        # check if goal reached
        dist_final = math.sqrt((robot.x - goal_node[0])**2 + (robot.y - goal_node[1])**2)
        
        if i % 20 == 0:
            # print log
            print(f"Step {i:<4}: Pos({robot.x:.1f}, {robot.y:.1f}) | v={u[0]:.2f} | GoalDist: {dist_final:.1f}")
            
        if dist_final < 0.5: # reached goal (within 0.5m tolerance)
            print(f"-"*30)
            print(f"REACHED GOAL AT STEP {i}!")
            reached = True
            break
            
    if not reached:
        print("TIMEOUT: Robot stuck or path too long.")

    # 5. visualization
    plt.figure(figsize=(10, 10))
    
    # plot map (note: imshow inverts y-axis)
    plt.imshow(map_obj.occupancy_map, cmap='gray', origin='upper')
    
    # plot planned path (green)
    plt.plot(global_path_smooth[:, 0], global_path_smooth[:, 1], 'g-', linewidth=2, label='Smoothed Path')
    
    # plot actual trajectory (red)
    plt.plot(history_x, history_y, 'r--', linewidth=2, label='Robot Traj')
    
    # plot start/goal points
    plt.plot(start_node[0], start_node[1], 'bo', label='Start')
    plt.plot(goal_node[0], goal_node[1], 'r*', markersize=15, label='Goal')
    
    plt.legend()
    plt.xlabel('X (Col)')
    plt.ylabel('Y (Row)')
    plt.title('Robot Navigation Result')
    
    plt.savefig('final_result_v2.png', dpi=300)
    print("Image saved: final_result_v2.png")
    plt.show()

if __name__ == '__main__':
    main()