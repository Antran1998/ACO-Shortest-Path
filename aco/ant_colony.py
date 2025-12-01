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
            self.actual_node = start_node_pos
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
            if self.actual_node == self.final_node:
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
            # Keep the first visited node (start), drop the rest
            if self.visited_nodes:
                self.visited_nodes[1:] = []
            else:
                self.remember_visited_node(self.start_pos)
            self.actual_node = self.start_pos

    def __init__(self, in_map, no_ants, iterations, evaporation_factor,
                 pheromone_adding_constant, initial_pheromone=1.0, alpha=1.0, beta=2.0, max_steps=None):
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.initial_pheromone = initial_pheromone
        self.alpha = alpha  # pheromone influence
        self.beta = beta    # heuristic influence
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
        '''Heuristic: inverse Manhattan distance to goal (closer nodes -> higher value) '''
        dy = abs(self.map.final_node[0] - dest_node[0])
        dx = abs(self.map.final_node[1] - dest_node[1])
        dist = dx + dy
        return 1.0 / (dist + 1.0)

    def select_next_node(self, actual_node, visited_nodes=None):
        ''' Randomly selects the next node using (pheromone^alpha)*(heuristic^beta) weighting.
            Prefer unvisited neighbors; if none available, allow visited ones as fallback. '''
        weights = []
        final_nodes = []
        # First try unvisited neighbors (reduces loops)
        for edge in actual_node.edges:
            if visited_nodes and edge['FinalNode'] in visited_nodes:
                continue
            pher = max(edge.get('Pheromone', 0.0), 0.0) ** self.alpha
            h = self._heuristic(edge['FinalNode']) ** self.beta
            weights.append(pher * h)
            final_nodes.append(edge['FinalNode'])
        # Fallback to any neighbor if none unvisited
        if not final_nodes:
            for edge in actual_node.edges:
                pher = max(edge.get('Pheromone', 0.0), 0.0) ** self.alpha
                h = self._heuristic(edge['FinalNode']) ** self.beta
                weights.append(pher * h)
                final_nodes.append(edge['FinalNode'])
        if not final_nodes:
            return actual_node.node_pos  # no moves available
        total = sum(weights)
        if total <= 0.0:
            # fallback to uniform random choice using integer index to avoid dtype issues
            idx = np.random.randint(len(final_nodes))
            return final_nodes[int(idx)]
        probs = [w / total for w in weights]
        idx = np.random.choice(len(final_nodes), p=probs)
        return final_nodes[int(idx)]

    def pheromone_update(self):
        ''' Updates the pheromone level of each trail (evaporation once + deposit from paths) '''
        # Evaporate once across all edges
        for row in self.map.nodes_array:
            for node in row:
                for edge in node.edges:
                    edge['Pheromone'] = max(edge.get('Pheromone', 0.0) * (1.0 - self.evaporation_factor), 1e-12)

        # Deposit pheromones for the discovered paths
        for path in self.paths:
            if not path:
                continue
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
        ''' Empty the list of paths '''
        self.paths = []

    def sort_paths(self):
        ''' Sorts the paths '''
        self.paths.sort(key=len)

    def add_to_path_results(self, in_path):
        ''' Appends the path to the results path list'''
        self.paths.append(in_path)

    def get_coincidence_indices(self, lst, element):
        ''' Gets the indices of the coincidences of elements in the path '''
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset + 1)
            except ValueError:
                return result
            result.append(offset)

    def delete_loops(self, in_path):
        ''' Checks if there is a loop in the resulting path and deletes it '''
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            coincidences.reverse()
            for i, coincidence in enumerate(coincidences):
                if not i == len(coincidences) - 1:
                    res_path[coincidences[i + 1]:coincidence] = []
        return res_path

    def calculate_path(self):
        ''' Carries out the process to get the best path (global best retained) '''
        for i in range(self.iterations):
            for ant in self.ants:
                ant.setup_ant()
                steps = 0
                while not ant.final_node_reached and steps < self.max_steps:
                    node_obj = self.map.nodes_array[int(ant.actual_node[0])][int(ant.actual_node[1])]
                    node_to_visit = self.select_next_node(node_obj, ant.get_visited_nodes())
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
