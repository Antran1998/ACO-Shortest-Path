import numpy as np
import random
import math

class AntColony:
    ''' Class used for handling the behaviour of the whole ant colony '''
    EPSILON = 1e-6  # Small value to avoid division by zero
    
    class Ant:
        ''' Class used for handling the ant's individual behaviour '''
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node = start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []
            self.visited_set = set()  # For O(1) lookup
            self.final_node_reached = False
            self.remember_visited_node(start_node_pos)
            
            # attributes for improve3 (division of labor)
            self.role = 'soldier'
            self.alpha = 1.0
            self.beta = 2.0

        def move_ant(self, node_to_visit):
            self.actual_node = node_to_visit
            self.remember_visited_node(node_to_visit)

        def remember_visited_node(self, node_pos):
            self.visited_nodes.append(node_pos)
            self.visited_set.add(node_pos)

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
                 destination_boost_radius=None,
                 boost_factor=3.0):
        
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
        
        # Detect if all features are active for parameter adjustment
        self.all_features_active = (use_cone_pheromone and 
                                    use_adaptive_processing and 
                                    use_division_of_labor)
        
        # improved params for cone pheromone (improve 1)
        map_dim = self.map.in_map.shape[0]
        # Increase radius for gentler gradient - use 40-50% of map dimension
        if destination_boost_radius is None:
            self.destination_boost_radius = max(8, int(map_dim * 0.45))
        else:
            self.destination_boost_radius = destination_boost_radius
        
        # When all features active, use even gentler cone influence
        if self.all_features_active:
            self.boost_factor = min(boost_factor, 1.0)  # Further reduced from 1.5
        else:
            self.boost_factor = min(boost_factor, 1.5)
        
        # Track current iteration for dynamic cone strength
        self.current_iteration = 0

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
        '''Calculate adaptive PHF (alpha) and EHF (beta) using adaptive processing.   
        '''      
        # Exact integral calculation: integral[0 to n/N] t dt = (n/N)^2/2
        integral_value = ((current_iteration / self.iterations) ** 2) / 2.0
        
        # Adaptive adjustment
        alpha_adaptive = self.base_alpha + self.xi * integral_value
        beta_adaptive = self.base_beta + self.xi * integral_value
        
        return alpha_adaptive, beta_adaptive

    def calculate_euclidean_distance(self, pos1, pos2):
        '''Calculate Euclidean distance between two positions.'''
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx**2 + dy**2)

    def calculate_path_distance(self, path):
        '''Calculate total Euclidean distance of a path.'''
        if len(path) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(len(path) - 1):
            distance = self.calculate_euclidean_distance(path[i+1], path[i])
            total_distance += distance
        return total_distance

    def initialize_pheromones(self):
        ''' Ensures every edge has an initial pheromone value.
            If use_cone_pheromone is True, creates a cone distribution. '''
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    pheromone_value = float(self.initial_pheromone)
                    
                    # improve 1: cone pheromone initialization with gentler gradient
                    if self.use_cone_pheromone:
                        dy = abs(self.map.final_node[0] - edge['FinalNode'][0])
                        dx = abs(self.map.final_node[1] - edge['FinalNode'][1])
                        dist = math.sqrt(dx**2 + dy**2)

                        if dist <= self.destination_boost_radius:
                            # Use gentler linear decay instead of exponential
                            # This provides guidance without overwhelming the search
                            normalized_dist = dist / self.destination_boost_radius
                            # Linear decay: 1.0 at goal, decreasing to 0 at radius edge
                            cone_boost = self.boost_factor * (1.0 - normalized_dist)
                            # Use additive boost instead of multiplicative to avoid extreme values
                            pheromone_value += cone_boost
                    
                    edge['Pheromone'] = pheromone_value

    def _heuristic(self, dest_node):
        '''Inverse Euclidean distance heuristic.'''
        distance = self.calculate_euclidean_distance(dest_node, self.map.final_node)
        return 1.0 / (distance + self.EPSILON)

    def select_next_node(self, actual_node, ant):
        ''' Selects next node using ant-specific alpha/beta and avoiding visited nodes '''
        weights = []
        final_nodes = []
        
        # First try: only unvisited nodes
        for edge in actual_node.edges:
            if edge['FinalNode'] not in ant.visited_set:
                pher = max(edge.get('Pheromone', 0.0), 0.0) ** ant.alpha
                h = self._heuristic(edge['FinalNode']) ** ant.beta
                
                weights.append(pher * h)
                final_nodes.append(edge['FinalNode'])
        
        # If all neighbors visited, allow backtracking with penalty
        if not final_nodes:
            for edge in actual_node.edges:
                pher = max(edge.get('Pheromone', 0.0), 0.0) ** ant.alpha
                h = self._heuristic(edge['FinalNode']) ** ant.beta
                # Heavy penalty for revisiting
                weights.append((pher * h) * 0.1)
                final_nodes.append(edge['FinalNode'])
        
        if not final_nodes:
            return actual_node.node_pos 

        total = sum(weights)
        if total <= 0.0:
            # Pure greedy heuristic fallback
            heuristics = [self._heuristic(node) for node in final_nodes]
            best_idx = np.argmax(heuristics)
            return final_nodes[best_idx]
            
        probs = [w / total for w in weights]
        idx = np.random.choice(len(final_nodes), p=probs)
        return final_nodes[int(idx)]

    def pheromone_update(self):
        ''' Updates the pheromone level - only evaporate edges on used paths '''
        self.sort_paths()
        
        # First: evaporate ALL edges globally
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    edge['Pheromone'] *= (1.0 - self.evaporation_factor)
        
        # Second: deposit pheromone on paths proportional to quality
        for i, path in enumerate(self.paths):
            path_distance = self.calculate_path_distance(path)
            if path_distance == 0:
                path_distance = self.EPSILON
            
            # Elite weighting: best paths get more pheromone
            # When all features active, reduce elite weighting to prevent over-exploitation
            if self.all_features_active:
                if i == 0:  # Best path
                    weight = 1.5
                elif i < len(self.paths) // 3:  # Top third
                    weight = 1.2
                else:
                    weight = 1.0
            else:
                if i == 0:  # Best path
                    weight = 2.0
                elif i < len(self.paths) // 3:  # Top third
                    weight = 1.5
                else:
                    weight = 1.0
                
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
        # Sort by Euclidean distance, not just node count
        self.paths.sort(key=lambda p: self.calculate_path_distance(p))

    def add_to_path_results(self, in_path):
        self.paths.append(in_path)

    def get_coincidence_indices(self, lst, element):
        ''' Gets the indices of the coincidences
            of elements in the path '''
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)

    def delete_loops(self, in_path):
        ''' Checks if there is a loop in the
            resulting path and deletes it '''
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            # reverse de list to delete elements from back to front of the list
            coincidences.reverse()
            for i, coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []

        return res_path

    def calculate_path(self):
        '''
        Calculate the optimal path using ACO with optional enhancements.
        
        Enhancement Order (when all enabled):
        1. Cone Pheromone: Pre-initialized bias toward destination (static)
        2. Adaptive Processing: Dynamically adjusts alpha/beta over iterations
        3. Division of Labor: Differentiates ant roles with adjusted multipliers
           - When combined with Adaptive Processing, uses moderate multipliers
           - When standalone, uses stronger multipliers
        
        This order ensures features complement each other rather than conflict.
        '''
        for i in range(self.iterations):
            self.current_iteration = i
            
            # improvement 2: calculate adaptive PHF and EHF if enabled
            if self.use_adaptive_processing:
                alpha_adaptive, beta_adaptive = self.calculate_adaptive_factors(i)
            else:
                alpha_adaptive, beta_adaptive = self.base_alpha, self.base_beta
            
            # improvement 3: division of labor with optimized parameters
            if self.use_division_of_labor:
                # Time factor: increases from 0 to 1
                S = (i + 1) / self.iterations
                
                # Quality metric based on path convergence
                if self.paths:
                    distances = [self.calculate_path_distance(p) for p in self.paths]
                    if distances:
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)
                        cv = std_dist / mean_dist if mean_dist > 0 else 1.0
                        # Convergence indicator: low CV = high convergence
                        theta = max(0.1, min(0.9, 1.0 - cv))
                    else:
                        theta = 0.5
                else:
                    # When cone pheromone is active, start with more exploitation
                    # since cone already provides good initial guidance
                    theta = 0.5 if self.use_cone_pheromone else 0.3
                
                # Transition function: sigmoid-like smooth transition
                Lambda = 1.0 / (1.0 + math.exp(-10.0 * (S - theta)))

                # Optimized role assignment with three scenarios:
                # 1. All features active: use gentlest multipliers
                # 2. Division + Adaptive: use moderate multipliers
                # 3. Division only: use stronger multipliers
                if self.all_features_active:
                    # Gentlest multipliers when all features combine
                    king_alpha_mult = 1.5
                    king_beta_mult = 0.8
                    soldier_alpha_mult = 0.8
                    soldier_beta_mult = 1.5
                elif self.use_adaptive_processing:
                    # Moderate multipliers when adaptive processing is active
                    king_alpha_mult = 2.0
                    king_beta_mult = 0.7
                    soldier_alpha_mult = 0.7
                    soldier_beta_mult = 1.8
                else:
                    # Stronger multipliers when adaptive processing is not used
                    king_alpha_mult = 3.0
                    king_beta_mult = 0.5
                    soldier_alpha_mult = 0.5
                    soldier_beta_mult = 2.5

                for ant in self.ants:
                    if np.random.random() < Lambda:
                        ant.role = 'king'
                        # King: exploit known good paths
                        ant.alpha = alpha_adaptive * king_alpha_mult
                        ant.beta = beta_adaptive * king_beta_mult
                    else:
                        ant.role = 'soldier'
                        # Soldier: explore with heuristic guidance
                        ant.alpha = alpha_adaptive * soldier_alpha_mult
                        ant.beta = beta_adaptive * soldier_beta_mult
            else:
                for ant in self.ants:
                    ant.role = 'normal'
                    ant.alpha = alpha_adaptive
                    ant.beta = beta_adaptive

            for ant in self.ants:
                ant.setup_ant()
                steps = 0
                while not ant.final_node_reached and steps < self.max_steps:
                    node_obj = self.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]
                    node_to_visit = self.select_next_node(node_obj, ant)
                    
                    # Check if stuck at same position
                    if node_to_visit == ant.actual_node:
                        break
                    
                    ant.move_ant(node_to_visit)
                    ant.is_final_node_reached()
                    steps += 1
                
                # Only add successful paths
                if ant.final_node_reached:
                    clean_path = self.delete_loops(ant.get_visited_nodes())
                    self.add_to_path_results(clean_path)
                
                ant.enable_start_new_path()

            # Update pheromones
            if self.paths:
                self.pheromone_update()
                self.best_result = self.paths[0]
                
                # Track global best
                best_distance = self.calculate_path_distance(self.paths[0])
                if best_distance < self.global_best_distance:
                    self.global_best_distance = best_distance
                    self.global_best_path = list(self.paths[0])
            
            self.empty_paths()
            
        # Return global best if available
        if self.global_best_path:
            return self.global_best_path
        return self.best_result