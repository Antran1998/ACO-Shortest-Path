import numpy as np
import random
import math

class AntColony:
    ''' Class used for handling the behaviour of the whole ant colony '''
    
    class Ant:
        ''' Class used for handling the ant's individual behaviour '''
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node = start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []
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
                 max_steps=None,
                 use_cone_pheromone=False,
                 use_division_of_labor=False,
                 destination_boost_radius=20,
                 boost_factor=6.0):
        
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.initial_pheromone = initial_pheromone
        
        self.base_alpha = alpha
        self.base_beta = beta
        
        self.use_cone_pheromone = use_cone_pheromone
        self.use_division_of_labor = use_division_of_labor
        
        # improved params for cone pheromone (improve 1)
        # adaptive radius based on map size
        map_dim = self.map.in_map.shape[0]
        self.destination_boost_radius = destination_boost_radius if destination_boost_radius > 0 else int(map_dim * 0.65)
        self.boost_factor = boost_factor

        if max_steps is None:
            dim = self.map.in_map.shape[0]
            self.max_steps = dim * dim 
        else:
            self.max_steps = max_steps
            
        self.paths = []
        self.ants = self.create_ants()
        self.best_result = []
        
        self.initialize_pheromones()

    def create_ants(self):
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.map.initial_node, self.map.final_node))
        return ants

    def initialize_pheromones(self):
        ''' Ensures every edge has an initial pheromone value.
            If use_cone_pheromone is True, creates a cone distribution. '''
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    pheromone_value = float(self.initial_pheromone)
                    
                    # improve 1: cone pheromone initialization with exponential decay
                    if self.use_cone_pheromone:
                        dy = abs(self.map.final_node[0] - edge['FinalNode'][0])
                        dx = abs(self.map.final_node[1] - edge['FinalNode'][1])
                        dist = math.sqrt(dx**2 + dy**2)

                        if dist <= self.destination_boost_radius:
                            # exponential decay for stronger concentration near goal
                            normalized_dist = dist / (self.destination_boost_radius + 1.0)
                            cone_boost = self.boost_factor * math.exp(-2.0 * normalized_dist)
                            pheromone_value *= (1.0 + cone_boost)
                    
                    edge['Pheromone'] = pheromone_value

    def _heuristic(self, dest_node):
        # Heuristic: 1 / (Distance + 1)
        dy = abs(self.map.final_node[0] - dest_node[0])
        dx = abs(self.map.final_node[1] - dest_node[1])
        dist = dx + dy
        return 1.0 / (dist + 1.0)

    def select_next_node(self, actual_node, ant, visited_nodes=None):
        ''' Selects next node using ant-specific alpha/beta '''
        weights = []
        final_nodes = []
        
        for edge in actual_node.edges:
            if visited_nodes and edge['FinalNode'] in visited_nodes:
                continue
            
            # use ant.alpha and ant.beta instead of global values
            pher = max(edge.get('Pheromone', 0.0), 0.0) ** ant.alpha
            h = self._heuristic(edge['FinalNode']) ** ant.beta
            
            weights.append(pher * h)
            final_nodes.append(edge['FinalNode'])
            
        # fallback if stuck
        if not final_nodes:
            for edge in actual_node.edges:
                pher = max(edge.get('Pheromone', 0.0), 0.0) ** ant.alpha
                h = self._heuristic(edge['FinalNode']) ** ant.beta
                weights.append(pher * h)
                final_nodes.append(edge['FinalNode'])
        
        if not final_nodes:
            return actual_node.node_pos 

        total = sum(weights)
        if total <= 0.0:
            idx = np.random.randint(len(final_nodes))
            return final_nodes[int(idx)]
            
        probs = [w / total for w in weights]
        idx = np.random.choice(len(final_nodes), p=probs)
        return final_nodes[int(idx)]

    def pheromone_update(self):
        # Evaporation
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    edge['Pheromone'] = max(edge.get('Pheromone', 0.0) * (1.0 - self.evaporation_factor), 1e-12)

        # Deposit
        for path in self.paths:
            if not path: continue
            deposit = self.pheromone_adding_constant / float(len(path))
            for j in range(len(path) - 1):
                current = path[j]
                nxt = path[j + 1]
                node = self.map.nodes_array[int(current[0])][int(current[1])]
                for edge in node.edges:
                    if edge['FinalNode'] == nxt:
                        edge['Pheromone'] += deposit
                        break

    def empty_paths(self):
        self.paths = []

    def sort_paths(self):
        self.paths.sort(key=len)

    def add_to_path_results(self, in_path):
        self.paths.append(in_path)

    def delete_loops(self, in_path):
        res_path = list(in_path)
        return res_path

    def calculate_path(self):
        for i in range(self.iterations):
            
            # improvement 3: division of labor with optimized parameters
            if self.use_division_of_labor:
                # 1. calculate s (time factor)
                S = (i + 1) / self.iterations
                
                # 2. improved theta calculation with weighted history
                path_lengths = [len(p) for p in self.paths] if self.paths else []
                if path_lengths:
                    mean_len = np.mean(path_lengths)
                    std_dev = np.std(path_lengths)
                    cv = std_dev / mean_len if mean_len > 0 else 0
                    # adaptive theta: more sensitive early, more stable later
                    theta = max(0.05, min(0.95, (1.0 - cv) * (0.5 + 0.5 * S)))
                else:
                    theta = 0.5 

                # 3. improved lambda with smoother transition
                Lambda = (S**2) / (S**2 + theta**2 + 0.01)

                # 4. assign roles with optimized alpha/beta ratios
                for ant in self.ants:
                    if np.random.random() < Lambda:
                        ant.role = 'king'
                        ant.alpha = 5.0  # king: strong pheromone preference
                        ant.beta = 0.8   # reduced heuristic weight
                    else:
                        ant.role = 'soldier'
                        ant.alpha = 0.8   # soldier: balanced exploration
                        ant.beta = 6.0    # strong heuristic preference
            else:
                # base aco: all ants use default params
                for ant in self.ants:
                    ant.role = 'normal'
                    ant.alpha = self.base_alpha
                    ant.beta = self.base_beta

            for ant in self.ants:
                ant.setup_ant()
                steps = 0
                while not ant.final_node_reached and steps < self.max_steps:
                    node_obj = self.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]
                    # important: pass 'ant' to access its specific alpha/beta
                    node_to_visit = self.select_next_node(node_obj, ant, ant.get_visited_nodes())
                    ant.move_ant(node_to_visit)
                    ant.is_final_node_reached()
                    steps += 1
                
                # simple loop removal (based on ant's visited list)
                # (using ant's list for simplicity, should use robust delete_loops in prod)
                self.add_to_path_results(ant.get_visited_nodes()) 
                ant.enable_start_new_path()

            self.pheromone_update()
            
            if self.paths:
                self.sort_paths()
                candidate = self.paths[0]
                if not self.best_result or len(candidate) < len(self.best_result):
                    self.best_result = candidate
            
            self.empty_paths()
            
        return self.best_result