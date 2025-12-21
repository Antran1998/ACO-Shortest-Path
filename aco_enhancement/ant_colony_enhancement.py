import numpy as np
import random
import math

class AntColony:
    ''' class used for handling the behaviour of the whole ant colony '''
    EPSILON = 1e-6  # small value to avoid division by zero
    
    class Ant:
        ''' class used for handling the ant's individual behaviour '''
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node = start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []
            self.visited_set = set()
            self.final_node_reached = False
            self.remember_visited_node(start_node_pos)
            
            # improvement 3: initialize default role
            self.role = 'normal' 
            self.alpha = 1.0
            self.beta = 2.0

            # IMPROVEMENT 4 (RENAMED): Advanced backtracking data structures
            self.decision_stack = []  # Stack of (node, unexplored_neighbors)
            self.backtrack_count = 0  # Track how many backtracks occurred
            self.max_backtracks = 10  # Limit backtracks per ant to prevent infinite loops

        def move_ant(self, node_to_visit):
            self.actual_node = node_to_visit
            self.remember_visited_node(node_to_visit)

        def remember_visited_node(self, node_pos):
            self.visited_nodes.append(node_pos)
            self.visited_set.add(node_pos)

        def save_decision_point(self, current_node, all_neighbors):
            """Save a decision point for potential backtracking"""
            # Only save if there are multiple choices
            unvisited = [n for n in all_neighbors if n not in self.visited_set]
            if len(unvisited) > 1:
                self.decision_stack.append({
                    'node': current_node,
                    'path_length': len(self.visited_nodes),
                    'unexplored': set(n for n in unvisited)
                })

        def backtrack_to_decision_point(self):
            """
            Backtrack to the most recent decision point with unexplored options.
            Returns the node to backtrack to, or None if no valid backtrack point.
            """
            while self.decision_stack:
                decision_point = self.decision_stack[-1]
                # Remove nodes we've already visited from unexplored set
                decision_point['unexplored'] -= self.visited_set
                # If this decision point has unexplored options, use it
                if decision_point['unexplored']:
                    # Restore path to this decision point
                    target_length = decision_point['path_length']
                    # Remove nodes after this decision point from visited set
                    nodes_to_remove = self.visited_nodes[target_length:]
                    for node in nodes_to_remove:
                        self.visited_set.discard(node)
                    # Truncate path
                    self.visited_nodes = self.visited_nodes[:target_length]
                    return decision_point['node']
                else:
                    # No unexplored options at this point, pop it
                    self.decision_stack.pop()
            return None  # No valid backtrack point found

        def calculate_path_quality(self, heuristic_func):
            """Calculate quality metric for current path direction"""
            if not self.visited_nodes or len(self.visited_nodes) < 2:
                return heuristic_func(self.actual_node)
            # Combine: distance to goal + path straightness
            goal_distance = heuristic_func(self.actual_node)
            # Penalize zigzagging (compare last 3 moves if available)
            straightness = 1.0
            if len(self.visited_nodes) >= 3:
                v1 = np.array(self.visited_nodes[-1]) - np.array(self.visited_nodes[-2])
                v2 = np.array(self.visited_nodes[-2]) - np.array(self.visited_nodes[-3])
                # Dot product normalized (1.0 = same direction, -1.0 = opposite)
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    straightness = 0.5 + 0.5 * np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return goal_distance * straightness

        def get_visited_nodes(self):
            return self.visited_nodes

        def is_final_node_reached(self):
            if self.actual_node == self.final_node:
                self.final_node_reached = True

        def enable_start_new_path(self):
            self.final_node_reached = False

        def setup_ant(self):
            if self.visited_nodes:
                self.visited_nodes[1:] = []
                self.visited_set.clear()
                self.visited_set.add(self.start_pos)
            else:
                self.remember_visited_node(self.start_pos)
            self.actual_node = self.start_pos

            # IMPROVEMENT 4: Reset backtracking structures
            self.decision_stack = []
            self.backtrack_count = 0

    def __init__(self, 
                 in_map, 
                 no_ants, 
                 iterations, 
                 evaporation_factor, 
                 pheromone_adding_constant, 
                 initial_pheromone=1.0, 
                 alpha=1.0, 
                 beta=2.0, 
                 xi=0.5,
                 max_steps=None,
                 use_cone_pheromone=False,
                 use_adaptive_processing=False,
                 use_division_of_labor=False,
                 use_backtracking=False):
        
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.initial_pheromone = initial_pheromone
        
        self.base_alpha = alpha
        self.base_beta = beta
        self.xi = xi
        
        self.use_cone_pheromone = use_cone_pheromone
        self.use_adaptive_processing = use_adaptive_processing
        self.use_division_of_labor = use_division_of_labor
        self.use_backtracking = use_backtracking  # RENAMED: improve 4
        
        # detect if all features are active for parameter adjustment
        self.all_features_active = (use_cone_pheromone and 
                                    use_adaptive_processing and 
                                    use_division_of_labor)
        
        # track current iteration for dynamic cone strength
        self.current_iteration = 0

        # IMPROVEMENT 4: Initialize global tabu table for persistent deadlocks
        self.tabu_list = set()

        # IMPROVEMENT 4: Backtracking statistics
        self.backtrack_stats = {
            'total_backtracks': 0,
            'successful_backtracks': 0,
            'failed_backtracks': 0
        }

        if max_steps is None:
            dim = self.map.in_map.shape[0]
            self.max_steps = dim * dim 
        else:
            self.max_steps = max_steps
            
        self.paths = []
        self.ants = self.create_ants()
        self.best_result = []
        self.global_best_path = None
        self.global_best_distance = float('inf')
        
        self.initialize_pheromones()

    def create_ants(self):
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.map.initial_node, self.map.final_node))
        return ants

    def calculate_adaptive_factors(self, current_iteration):
        ''' calculate adaptive phf (alpha) and ehf (beta) using adaptive processing '''      
        # exact integral calculation: integral[0 to n/N] t dt = (n/N)^2/2
        integral_value = ((current_iteration / self.iterations) ** 2) / 2.0
        
        # adaptive adjustment
        alpha_adaptive = self.base_alpha + self.xi * integral_value
        beta_adaptive = self.base_beta + self.xi * integral_value
        
        return alpha_adaptive, beta_adaptive

    def calculate_euclidean_distance(self, pos1, pos2):
        ''' calculate euclidean distance between two positions '''
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx**2 + dy**2)

    def calculate_path_distance(self, path):
        ''' calculate total euclidean distance of a path '''
        if len(path) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(len(path) - 1):
            distance = self.calculate_euclidean_distance(path[i+1], path[i])
            total_distance += distance
        return total_distance

    def initialize_pheromones(self):
        ''' ensures every edge has an initial pheromone value '''
        map_len = self.map.in_map.shape[0]
        
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    if self.use_cone_pheromone:
                        # get node coordinates
                        node_x = edge['FinalNode'][1]  # horizontal coordinate
                        node_y = edge['FinalNode'][0]  # vertical coordinate
                        
                        # calculate euclidean distance to target
                        dy = abs(self.map.final_node[0] - edge['FinalNode'][0])
                        dx = abs(self.map.final_node[1] - edge['FinalNode'][1])
                        d = math.sqrt(dx**2 + dy**2)
                        
                        # improve 1: cone pheromone formula
                        coordinate_diff = abs(node_x - node_y)
                        cone_term = (0.09 * coordinate_diff) / map_len
                        inverse_dist_term = 1.0 / (d + self.EPSILON)
                        
                        pheromone_value = cone_term + inverse_dist_term
                        pheromone_value = self.initial_pheromone + pheromone_value
                    else:
                        pheromone_value = float(self.initial_pheromone)
                    
                    edge['Pheromone'] = pheromone_value
                    edge['Probability'] = 0.0

    def _heuristic(self, dest_node):
        ''' inverse euclidean distance heuristic '''
        distance = self.calculate_euclidean_distance(dest_node, self.map.final_node)
        return 1.0 / (distance + self.EPSILON)

    def select_next_node(self, actual_node, ant):
        '''
        Selects next node based on role-specific logic with intelligent backtracking.
        IMPROVEMENT 4: Replaces simple deadlock recovery with multi-step backtracking.
        '''
        weights = []
        final_nodes = []
        
        # Get all neighbors for decision point tracking
        all_neighbors = [edge['FinalNode'] for edge in actual_node.edges]

        # IMPROVEMENT 4: Save decision point if multiple choices exist
        if self.use_backtracking and len(all_neighbors) > 1:
            ant.save_decision_point(ant.actual_node, all_neighbors)

        # Filter valid candidates (unvisited nodes, not in tabu list)
        candidates = []
        for edge in actual_node.edges:
            # IMPROVEMENT 4: Check global tabu list (for persistent problematic nodes)
            if self.use_backtracking and edge['FinalNode'] in self.tabu_list:
                continue
                
            if edge['FinalNode'] not in ant.visited_set:
                candidates.append(edge)

        # IMPROVEMENT 4: Multi-step intelligent backtracking when stuck
        if not candidates:
            if self.use_backtracking and ant.backtrack_count < ant.max_backtracks:
                backtrack_node = ant.backtrack_to_decision_point()
                if backtrack_node is not None:
                    ant.backtrack_count += 1
                    self.backtrack_stats['total_backtracks'] += 1
                    return backtrack_node
                else:
                    # No valid backtrack point - this is a true deadlock
                    self.backtrack_stats['failed_backtracks'] += 1
                    self.tabu_list.add(ant.actual_node)  # Mark as problematic
                    return None
            elif not self.use_backtracking:
                # Fallback: allow revisiting neighbors (old behavior)
                candidates = actual_node.edges
            else:
                # Exceeded backtrack limit or true deadlock
                return None

        if not candidates:
            return None  # Complete deadlock

        # soldier logic: epsilon-greedy exploration
        # use getattr to avoid attribute error
        if getattr(ant, 'role', 'normal') == 'soldier':
            # define rate explicitly for readability
            exploration_rate = 0.2
            
            # use numpy for probability check, random.choice for selection
            if candidates and np.random.random() < exploration_rate:
                return random.choice(candidates)['FinalNode']

        # standard aco logic (for kings, normal ants, or soldiers not exploring)
        for edge in candidates:
            # get pheromone and heuristic values
            pher = max(edge.get('Pheromone', 0.0), 0.0) ** ant.alpha
            h = self._heuristic(edge['FinalNode']) ** ant.beta
            # IMPROVEMENT 4: Adaptive penalty based on path quality
            if edge['FinalNode'] not in ant.visited_set:
                penalty = 1.0
            else:
                # Dynamic backtracking penalty (0.05 to 0.2 range)
                if self.use_backtracking:
                    current_quality = ant.calculate_path_quality(self._heuristic)
                    penalty = 0.05 + 0.15 * current_quality
                else:
                    penalty = 0.1  # Fixed penalty (old behavior)
            
            weights.append(pher * h * penalty)
            final_nodes.append(edge['FinalNode'])
        
        # safety check for zero weights
        total = sum(weights)
        if total <= 0.0:
            # fallback: choose nearest node purely by heuristic
            heuristics = [self._heuristic(node) for node in final_nodes]
            if not heuristics: return actual_node.node_pos # stuck
            best_idx = np.argmax(heuristics)
            return final_nodes[best_idx]
            
        # roulette wheel selection
        probs = [w / total for w in weights]
        idx = np.random.choice(len(final_nodes), p=probs)
        # IMPROVEMENT 4: Track successful backtracks
        selected_node = final_nodes[int(idx)]
        if self.use_backtracking and ant.backtrack_count > 0 and selected_node not in ant.visited_set:
            self.backtrack_stats['successful_backtracks'] += 1
        return selected_node

    def pheromone_update(self):
        ''' updates the pheromone level - only evaporate edges on used paths '''
        self.sort_paths()
        
        # first: evaporate all edges globally
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    edge['Pheromone'] *= (1.0 - self.evaporation_factor)
        
        # second: deposit pheromone on paths proportional to quality
        for i, path in enumerate(self.paths):
            path_distance = self.calculate_path_distance(path)
            if path_distance == 0:
                path_distance = self.EPSILON
            
            # elite weighting: best paths get more pheromone
            if self.all_features_active:
                if i == 0: weight = 1.5
                elif i < len(self.paths) // 3: weight = 1.2
                else: weight = 1.0
            else:
                if i == 0: weight = 2.0
                elif i < len(self.paths) // 3: weight = 1.5
                else: weight = 1.0
                
            pheromone_deposit = weight * self.pheromone_adding_constant / path_distance
            
            for j in range(len(path) - 1):
                element = path[j]
                next_element = path[j + 1]
                node = self.map.nodes_array[int(element[0])][int(element[1])]
                
                for edge in node.edges:
                    if edge['FinalNode'] == next_element:
                        edge['Pheromone'] += pheromone_deposit
                        break

    def empty_paths(self):
        self.paths = []

    def sort_paths(self):
        # sort by euclidean distance
        self.paths.sort(key=lambda p: self.calculate_path_distance(p))

    def add_to_path_results(self, in_path):
        self.paths.append(in_path)

    def get_coincidence_indices(self, lst, element):
        ''' gets the indices of the coincidences '''
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)

    def delete_loops(self, in_path):
        ''' deletes loops from the path '''
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            coincidences.reverse()
            for i, coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []
        return res_path

    def calculate_path(self):
        '''
        execute optimized aco flow (improvement 6)
        combines cone pheromone, adaptive factors, division of labor, and deadlock recovery
        '''
        # IMPROVEMENT 4: Reset tabu table and backtracking stats at start
        if self.use_backtracking:
            self.tabu_list.clear()
            self.backtrack_stats = {
                'total_backtracks': 0,
                'successful_backtracks': 0,
                'failed_backtracks': 0
            }

        for i in range(self.iterations):
            self.current_iteration = i
            
            # improvement 2: adaptive processing
            if self.use_adaptive_processing:
                alpha_adaptive, beta_adaptive = self.calculate_adaptive_factors(i)
            else:
                alpha_adaptive, beta_adaptive = self.base_alpha, self.base_beta
            
            # improvement 3: division of labor
            if self.use_division_of_labor:
                # time factor S
                S = 2.0 * ((i + 1) / self.iterations)
                
                # calculate theta
                if self.paths:
                    path_distances = [self.calculate_path_distance(p) for p in self.paths]
                    if len(path_distances) > 1:
                        mean_dist = np.mean(path_distances)
                        std_dist = np.std(path_distances)
                        cv = std_dist / (mean_dist + self.EPSILON)
                        theta = max(0.1, min(0.9, 1.0 - cv))
                    else:
                        theta = 0.5
                else:
                    theta = 0.5

                # transition probability
                Lambda = (S**2) / (S**2 + theta**2)

                # assign roles
                for ant in self.ants:
                    if np.random.random() < Lambda:
                        ant.role = 'king'
                        ant.alpha = alpha_adaptive * 1.5 
                        ant.beta = beta_adaptive
                    else:
                        ant.role = 'soldier'
                        ant.alpha = alpha_adaptive
                        ant.beta = beta_adaptive
            else:
                for ant in self.ants:
                    ant.role = 'normal'
                    ant.alpha = alpha_adaptive
                    ant.beta = beta_adaptive

            # main ant movement loop
            for ant in self.ants:
                ant.setup_ant()
                steps = 0
                deadlock_detected = False
                
                while not ant.final_node_reached and steps < self.max_steps:
                    node_obj = self.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]
                    
                    # try to select next node
                    node_to_visit = self.select_next_node(node_obj, ant)

                    # IMPROVEMENT 4: Deadlock handling with backtracking
                    if node_to_visit is None:
                        deadlock_detected = True
                        break
                    
                    # standard move
                    if node_to_visit == ant.actual_node:
                        break
                    
                    ant.move_ant(node_to_visit)
                    ant.is_final_node_reached()
                    steps += 1
                
                # Only store path if reached goal without deadlock
                if ant.final_node_reached and not deadlock_detected:
                    clean_path = self.delete_loops(ant.get_visited_nodes())
                    self.add_to_path_results(clean_path)
                
                ant.enable_start_new_path()

            # pheromone update
            # improve 6: update only after all ants finished searching
            if self.paths:
                self.pheromone_update()
                
                # update global best
                current_best = self.paths[0]
                current_dist = self.calculate_path_distance(current_best)
                
                if current_dist < self.global_best_distance:
                    self.global_best_distance = current_dist
                    self.global_best_path = list(current_best)
            
            # reset paths for next iteration
            self.empty_paths()

        # IMPROVEMENT 4: Print backtracking statistics
        # if self.use_backtracking:
        #     total = self.backtrack_stats['total_backtracks']
        #     successful = self.backtrack_stats['successful_backtracks']
        #     failed = self.backtrack_stats['failed_backtracks']
        #     success_rate = (successful / total * 100) if total > 0 else 0
        #     print(f"[Backtracking Stats] Total: {total}, Successful: {successful}, "
        #           f"Failed: {failed}, Success Rate: {success_rate:.1f}%")
            
        # return the best path found across all iterations
        if self.global_best_path:
            return self.global_best_path
        return self.best_result

