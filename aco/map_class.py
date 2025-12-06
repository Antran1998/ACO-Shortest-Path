#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class Map:
    ''' Class used for handling the
        information provided by the
        input map '''
    class Nodes:
        ''' Class for representing the nodes
            used by the ACO algorithm '''
        def __init__(self, row, col, in_map, spec):
            self.node_pos = (row, col)
            self.edges = self.compute_edges(in_map)
            self.spec = spec

        def compute_edges(self, map_arr):
            ''' class that returns the edges
                connected to each node '''
            imax = map_arr.shape[0]
            jmax = map_arr.shape[1]
            edges = []
            if map_arr[self.node_pos[0]][self.node_pos[1]] == 1:
                for dj in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        newi = self.node_pos[0] + di
                        newj = self.node_pos[1] + dj
                        if (dj == 0 and di == 0):
                            continue
                        if (newj >= 0 and newj < jmax and newi >= 0 and newi < imax):
                            if map_arr[newi][newj] == 1:
                                # Only store topology - let algorithm handle state
                                edges.append({'FinalNode': (newi, newj)})
            return edges

    def __init__(self, map_name):
        self.in_map = self._read_map(map_name)
        self.occupancy_map = self._map_2_occupancy_map()
        s_pos = np.argwhere(self.in_map == 'S')
        f_pos = np.argwhere(self.in_map == 'F')
        if s_pos.size == 0 or f_pos.size == 0:
            raise ValueError("Map must contain exactly one 'S' and one 'F' marker")
        self.initial_node = (int(s_pos[0][0]), int(s_pos[0][1]))
        self.final_node = (int(f_pos[0][0]), int(f_pos[0][1]))
        self.nodes_array = self._create_nodes()

    def _create_nodes(self):
        ''' Create nodes out of the initial map '''
        rows = self.in_map.shape[0]
        cols = self.in_map.shape[1]
        return [[self.Nodes(i, j, self.occupancy_map, self.in_map[i][j]) for j in range(cols)] for i in range(rows)]

    def _read_map(self, map_name):
        ''' Reads data from an input map txt file'''
        map_planning = np.loadtxt('./maps/' + map_name, dtype=str)
        return map_planning

    def _map_2_occupancy_map(self):
        ''' Takes the matrix and converts it into a int occupancy array (1=free, 0=obstacle) '''
        occ = np.zeros(self.in_map.shape, dtype=int)
        occ[self.in_map == 'E'] = 1
        occ[self.in_map == 'S'] = 1
        occ[self.in_map == 'F'] = 1
        # 'O' remains 0
        return occ

    def represent_map(self):
        ''' Represents the map '''
        # Map representation
        plt.plot(self.initial_node[1], self.initial_node[0], 'ro', markersize=10)
        plt.plot(self.final_node[1], self.final_node[0], 'bo', markersize=10)
        plt.imshow(self.occupancy_map, cmap='gray', interpolation='nearest')
        plt.show()
        plt.close()

    def represent_path(self, path):
        ''' Represents the path in the map '''
        x = []
        y = []
        for p in path:
            x.append(p[1])
            y.append(p[0])
        plt.plot(x, y)
        self.represent_map()
