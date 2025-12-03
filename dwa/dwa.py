import math
import numpy as np

class DWAConfig:
    def __init__(self):
        # kinematic limits (table 1)
        self.max_speed = 1.0        # [m/s]
        self.min_speed = 0.0        # prevent backward motion
        
        # maximum selection speed (rotational speed): 0.56 rad/s
        self.max_yaw_rate = 0.56    
        
        # acceleration: 0.25 m/s^2
        self.max_accel = 0.25       
        self.max_decel = 0.25
        
        # rotational acceleration: 0.85 rad/s^2
        self.max_delta_yaw_rate = 0.85 
        
        # resolution: v=0.01 m/s, w=0.02 rad/s
        self.v_resolution = 0.01    
        self.yaw_rate_resolution = 0.02 * (math.pi/180)
        
        # simulation & safety
        self.dt = 0.1
        self.predict_time = 2.0     # tune for map (paper doesn't fix this)
        
        # safe distance: 0.35 m
        self.radius = 0.35          
        
        # weights (evaluation function)
        # "0.45 for path tracking, 0.35 for obstacle avoidance, 0.12 for speed, 0.08 for heading"
        self.w_path_dir = 0.45       # most important (tracks global path)
        self.w_dist = 0.35           # obstacle avoidance
        self.w_vel = 0.12            # velocity
        self.w_target_heading = 0.08 # heading to goal

class RobotState:
    def __init__(self, x, y, theta, v, w):
        self.x = x; self.y = y; self.theta = theta; self.v = v; self.w = w

def motion_model(state, v, w, dt):
    state.theta += w * dt
    state.x += v * math.cos(state.theta) * dt
    state.y += v * math.sin(state.theta) * dt
    state.v = v; state.w = w
    return state

def calc_dynamic_window(state, config):
    # 1. physical limits
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
    
    # 2. dynamic limits (acceleration)
    Vd = [state.v - config.max_decel * config.dt, state.v + config.max_accel * config.dt,
          state.w - config.max_delta_yaw_rate * config.dt, state.w + config.max_delta_yaw_rate * config.dt]
    
    # intersection of Vs and Vd
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

def improved_dwa_control(state, config, goal, obstacles, global_path):
    dw = calc_dynamic_window(state, config)
    best_u = [0.0, 0.0]
    best_score = -float('inf')
    best_traj = np.array([state.x, state.y, state.theta])

    # iterate velocity space
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for w in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            
            # improve 2: dynamic velocity sampling (Va)
            # simulate trajectory then calculate
            # find min distance to obstacles on predicted path
            
            # 1. predict trajectory
            traj = np.array([state.x, state.y, state.theta])
            curr_state = RobotState(state.x, state.y, state.theta, state.v, state.w)
            
            for _ in range(int(config.predict_time / config.dt)):
                curr_state = motion_model(curr_state, v, w, config.dt)
                traj = np.vstack((traj, [curr_state.x, curr_state.y, curr_state.theta]))

            # 2. calc min distance to obstacle
            min_r = float('inf')
            if len(obstacles) > 0:
                # distance from all path points to all obstacles
                # sampling for optimization: use endpoints or sparse points
                diff = obstacles[:, np.newaxis, :] - traj[:, :2] 
                dists = np.linalg.norm(diff, axis=2)
                min_r = np.min(dists) - config.radius # minus robot radius

            # check safety condition Va (Eq 10) 
            # v <= sqrt(2 * dist * max_decel)
            # skip if current velocity exceeds safe braking distance
            safe_velocity = math.sqrt(2 * max(0, min_r) * config.max_decel)
            
            if v > safe_velocity: 
                continue # violates improve 2: unsafe

            if min_r <= 0: # collision detected
                continue

            # calculate objective function score G'
            
            # 1. H: Target Heading (towards final goal)
            dx = goal[0] - traj[-1, 0]
            dy = goal[1] - traj[-1, 1]
            target_angle = math.atan2(dy, dx)
            heading_score = 180 - math.degrees(abs(target_angle - traj[-1, 2]))

            # 2. D: Distance (min_r calculated above)
            dist_score = min_r # further from obstacles is better

            # 3. V: Velocity
            vel_score = v

            # 4. Improve 1: G_head: Path Direction
            g_head_score = 0
            if global_path is not None:
                # find nearest point on global path
                dists_to_path = np.linalg.norm(global_path - traj[-1, :2], axis=1)
                min_idx = np.argmin(dists_to_path)
                
                # calc tangent angle of global path at that point
                # use next point for direction vector
                if min_idx < len(global_path) - 1:
                    p_curr = global_path[min_idx]
                    p_next = global_path[min_idx+1]
                    path_angle = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
                    
                    # calc angular deviation between robot and path
                    angle_diff = math.degrees(abs(path_angle - traj[-1, 2]))
                    # normalize to [0, 180]
                    angle_diff = (angle_diff + 180) % 360 - 180
                    
                    # Eq (5): Ghead = 180 - |theta|
                    g_head_score = 180 - abs(angle_diff)

            # aggregate G' score
            final_score = (config.w_target_heading * heading_score +
                           config.w_dist * dist_score +
                           config.w_vel * vel_score +
                           config.w_path_dir * g_head_score)

            if final_score > best_score:
                best_score = final_score
                best_u = [v, w]
                best_traj = traj

    return best_u, best_traj