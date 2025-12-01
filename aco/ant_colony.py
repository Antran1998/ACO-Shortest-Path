#!/usr/bin/env python

import numpy as np

class AntColony:
    ''' Class used for handling
        the behaviour of the whole
        ant colony '''
    class Ant:
        ''' Class used for handling
            the ant's
            individual behaviour '''
        def __init__(self, start_node_pos, final_node_pos):
            self.start_pos = start_node_pos
            self.actual_node= start_node_pos
            self.final_node = final_node_pos
            self.visited_nodes = []
            self.final_node_reached = False
            self.remember_visited_node(start_node_pos)

        def move_ant(self, node_to_visit):
            ''' Moves ant to the selected node '''
            self.actual_node = node_to_visit
            self.remember_visited_node(node_to_visit)

        def remember_visited_node(self, node_pos):
            ''' Appends the visited node to
                the list of visited nodes '''
            self.visited_nodes.append(node_pos)

        def get_visited_nodes(self):
            ''' Returns the list of
                visited nodes '''
            return self.visited_nodes

        def is_final_node_reached(self):
            ''' Checks if the ant has reached the
                final destination '''
            if self.actual_node == self.final_node :
                self.final_node_reached = True

        def enable_start_new_path(self):
            ''' Enables a new path search setting
                the final_node_reached variable to
                false '''
            self.final_node_reached = False

        def setup_ant(self):
            ''' Clears the list of visited nodes
                it stores the first one
                and selects the first one as initial'''
            self.visited_nodes[1:] =[]
            self.actual_node= self.start_pos

    def __init__(self, in_map, no_ants, iterations, evaporation_factor,
                 pheromone_adding_constant, initial_pheromone=1.0, alpha=1.0, beta=2.0,
                 override_initial_pheromone=False, max_steps=None):
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.initial_pheromone = initial_pheromone
        self.alpha = alpha  # pheromone influence
        self.beta = beta    # heuristic influence
        self.override_initial_pheromone = override_initial_pheromone
        # Max steps safeguard per ant to avoid infinite loops
        if max_steps is None:
            dim = self.map.in_map.shape[0]
            self.max_steps = dim * dim  # square of side length
        else:
            self.max_steps = max_steps
        self.paths = []
        self.ants = self.create_ants()
        self.best_result = []
        self.initialize_pheromones()

    def initialize_pheromones(self):
        ''' Ensures every edge has an initial pheromone value; override if requested '''
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    if self.override_initial_pheromone:
                        edge['Pheromone'] = float(self.initial_pheromone)
                    else:
                        if 'Pheromone' not in edge or edge['Pheromone'] <= 0.0:
                            edge['Pheromone'] = float(self.initial_pheromone)

    def create_ants(self):
        ''' Creates a list containin the
            total number of ants specified
            in the initial node '''
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.map.initial_node, self.map.final_node))
        return ants

    def _heuristic(self, dest_node):
        ''' Heuristic: inverse Manhattan distance to goal (closer nodes -> higher value) '''
        dy = abs(self.map.final_node[0] - dest_node[0])
        dx = abs(self.map.final_node[1] - dest_node[1])
        dist = dx + dy
        return 1.0 / (dist + 1.0)

    def select_next_node(self, actual_node):
        ''' Randomly selects the next node using (pheromone^alpha)*(heuristic^beta) weighting '''
        weights = []
        final_nodes = []
        for edge in actual_node.edges:
            pher = max(edge['Pheromone'], 0.0) ** self.alpha
            h = self._heuristic(edge['FinalNode']) ** self.beta
            weights.append(pher * h)
            final_nodes.append(edge['FinalNode'])
        total = sum(weights)
        if total <= 0.0:
            # Fallback to uniform random choice if all pheromones are zero
            return np.random.choice(final_nodes, 1)[0]
        probs = [w/total for w in weights]
        return np.random.choice(final_nodes, 1, p=probs)[0]

    def pheromone_update(self):
        ''' Updates the pheromone level of each trail (evaporation + deposit) '''
        self.sort_paths()
        for path in self.paths:
            for j, element in enumerate(path):
                for edge in self.map.nodes_array[element[0]][element[1]].edges:
                    if (j+1) < len(path):
                        if edge['FinalNode'] == path[j+1]:
                            edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone'] + \
                                self.pheromone_adding_constant/float(len(path))
                        else:
                            edge['Pheromone'] = (1.0 - self.evaporation_factor) * edge['Pheromone']

    def empty_paths(self):
        ''' Empty the list of paths '''
        self.paths = []

    def sort_paths(self):
        ''' Sorts the paths '''
        self.paths.sort(key=len)

    def add_to_path_results(self, in_path):
        ''' Appends the path to the results path list'''
        self.paths.append(in_path)

    def get_coincidence_indices(self,lst, element):
        ''' Gets the indices of the coincidences of elements in the path '''
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)

    def delete_loops(self, in_path):
        ''' Checks if there is a loop in the resulting path and deletes it '''
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            coincidences.reverse()
            for i,coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []
        return res_path

    def calculate_path(self):
        ''' Carries out the process to get the best path (global best retained) '''
        for i in range(self.iterations):
            for ant in self.ants:
                ant.setup_ant()
                steps = 0
                while not ant.final_node_reached and steps < self.max_steps:
                    node_to_visit = self.select_next_node(self.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])])
                    ant.move_ant(node_to_visit)
                    ant.is_final_node_reached()
                    steps += 1
                self.add_to_path_results(self.delete_loops(ant.get_visited_nodes()))
                ant.enable_start_new_path()
            # Update pheromones based on all paths this iteration
            self.pheromone_update()
            # Update global best only if improved
            if self.paths:
                self.sort_paths()
                candidate = self.paths[0]
                if not self.best_result or len(candidate) < len(self.best_result):
                    self.best_result = candidate
            self.empty_paths()
            print('Iteration: ', i, ' length of current best path: ', len(self.best_result))
        return self.best_result
