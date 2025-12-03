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
            self.role = 'soldier' # default role
            self.alpha = 1.0      # Pheromone weight
            self.beta = 2.0       # Heuristic weight

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
                 use_cone_pheromone=False,      # toggle improve 1 (cone pheromone)
                 use_division_of_labor=False,   # toggle improve 3 (division of labor)
                 destination_boost_radius=15,   # param for improve 1
                 boost_factor=4.0):             # param for improve 1
        
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.initial_pheromone = initial_pheromone
        
        # default values (improve 3 off)
        self.base_alpha = alpha
        self.base_beta = beta
        
        # save toggle configuration
        self.use_cone_pheromone = use_cone_pheromone
        self.use_division_of_labor = use_division_of_labor
        
        # params for cone pheromone (improve 1)
        self.destination_boost_radius = destination_boost_radius
        self.boost_factor = boost_factor

        if max_steps is None:
            dim = self.map.in_map.shape[0]
            self.max_steps = dim * dim 
        else:
            self.max_steps = max_steps
            
        self.paths = []
        self.ants = self.create_ants()
        self.best_result = []
        
        # init pheromones (improve 1 on)
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
                    
                    # improve 1: cone pheromone initialization
                    if self.use_cone_pheromone:
                        # calculate manhattan distance to target
                        dy = abs(self.map.final_node[0] - edge['FinalNode'][0])
                        dx = abs(self.map.final_node[1] - edge['FinalNode'][1])
                        dist = dx + dy

                        if dist <= self.destination_boost_radius:
                            # closer to target means denser pheromone (cone shape)
                            cone_boost = self.boost_factor * (1.0 - (dist / (self.destination_boost_radius + 1.0)))
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
        # print header if running standalone, but main prints during benchmark
        # print(f"{'Iter':<5} | {'S':<5} | {'Lmb':<5} | {'Soldiers':<3}/{'Kings':<3}")

        for i in range(self.iterations):
            
            # improvement 3: division of labor
            if self.use_division_of_labor:
                # 1. calculate s (time factor)
                S = (i + 1) / self.iterations
                
                # 2. calculate theta (sensitivity)
                path_lengths = [len(p) for p in self.paths] if self.paths else []
                if path_lengths:
                    mean_len = np.mean(path_lengths)
                    std_dev = np.std(path_lengths)
                    cv = std_dev / mean_len if mean_len > 0 else 0
                    theta = max(0.01, min(0.99, 1.0 - cv))
                else:
                    theta = 0.5 

                # 3. calculate lambda (transition prob)
                Lambda = (S**2) / (S**2 + theta**2 + 1e-9)

                # 4. assign roles
                for ant in self.ants:
                    if np.random.random() < Lambda:
                        ant.role = 'king'
                        ant.alpha = 4.0 # king: prefers pheromone
                        ant.beta = 1.0
                    else:
                        ant.role = 'soldier'
                        ant.alpha = 1.0 # soldier: prefers heuristic
                        ant.beta = 5.0
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