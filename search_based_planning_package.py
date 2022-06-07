import numpy as np
import networkx as nx
import itertools



class Planner():
    def __init__(self):
        '''
        here the planner converts a 2d numpy array consisting of 1's and 0's (1 - occupied by obstacle, 0 -free),
        to a graph with each node connected to all of its 8 neighbours
        '''
        self.grid = np.zeros((10, 10))
        self.nodes = self.grid.shape

        self.G = nx.grid_2d_graph(self.nodes[0], self.nodes[1])
        self.G.remove_edges_from(self.G.edges)

        ept_edge = []

        for i in range(self.nodes[0]):

            for j in range(self.nodes[1]):

                if (i+1) < self.nodes[0]:
                    ept_edge.append(((i, j), (i+1, j)))


                if 0 <= (i-1) < self.nodes[0] :
                    ept_edge.append(((i, j), (i-1, j)))


                if (j+1) < self.nodes[1] :
                    ept_edge.append(((i, j), (i, j+1)))

                if 0 <= (j-1) < self.nodes[1] :
                    ept_edge.append(((i, j), (i, j-1)))

                # Diagonilsation

                # if (i+1) < self.nodes[0] and (j+1) < self.nodes[1] :
                #     ept_edge.append(((i, j), (i+1, j+1)))

                # if (i+1) < self.nodes[0] and 0 <= (j-1) < self.nodes[1] :
                #     ept_edge.append(((i, j), (i+1, j-1)))

                # if 0 <= (i-1) < self.nodes[0] and (j+1) < self.nodes[1] :
                #     ept_edge.append(((i, j), (i-1, j+1)))

                # if 0 <= (i-1) < self.nodes[0] and 0 <= (j-1) < self.nodes[1] :
                #     ept_edge.append(((i, j), (i-1, j-1)))


        
        self.G.add_edges_from(ept_edge)

        self.deleted_nodes = 0  # counter to keep track of deleted nodes

        nx.set_edge_attributes(self.G, 1, name='cost')

    def delete_node(self):
        '''
        In case a node is an obstacle and has yet not been delted from the graph, then delete the node from the central graph.
        '''
        for i in range(self.nodes[0]):
            for j in range(self.nodes[1]):
                if (self.grid[i, j] == 1) and ((i, j) in self.G.nodes):
                    self.G.remove_node((i, j))
                    self.deleted_nodes += 1

                    # Diagnolisation
                    # data = list(itertools.permutations([ (i+1, j) , (i-1, j) , (i, j+1) , (i,j-1) ],2))
                    # self.G.remove_edges_from(data)

    def compute_path(self, start, goal, RESOLUTION):
        '''
        calculaes astar path between start and end nodes 
        '''
        final_path = []
        astar_path = nx.astar_path(
            self.G, start, goal, heuristic=euclidean, weight='cost')

        for i in range(len(astar_path)-1):
            a = astar_path[i]
            b = astar_path[i+1]
            temp_path = []
            if i == 0:
                temp_path.append(a)
            """
            Initialize a RESOLUTION variable to be changed according to the path
            Example resolution=1 means wp->(1,2,3)
                    resolution=2 means wp->(1,1.5,2)
            """
            for j in range(1, RESOLUTION):
                x_new = a[0] + ((b[0] - a[0]) * j / RESOLUTION)
                y_new = a[1] + ((b[1] - a[1]) * j / RESOLUTION)
                temp_path.append((x_new, y_new))
            temp_path.append(b)
            final_path.extend(temp_path)

        if len(final_path) == 0:
            final_path.append(start)
        return final_path


def euclidean(node1, node2):  # Euclidean function that takes in the node(x, y) and compute the distance
    x1, y1 = node1
    x2, y2 = node2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
