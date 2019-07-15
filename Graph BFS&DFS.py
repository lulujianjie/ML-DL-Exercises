
# coding: utf-8

# In[1]:


graph = {
    'A':['B','C'],
    'B':['A','C','D'],
    'C':['A','B','D','E'],
    'D':['B','C','E','F'],
    'E':['C','D'],
    'F':['D']
}

import numpy as np
def Graph2AdjMatrix(graph):
    length = len(graph.keys())
    indices = {v:k for k,v in enumerate(graph.keys())}
    A = np.zeros((length, length))
    for k in graph.keys():
        i = indices[k]
        neighbors = graph[k]
        for node in neighbors:
            j = indices[node]
            A[i][j] = 1
    return A, graph.keys()
A, keys = Graph2AdjMatrix(graph)
print(A, keys)


# In[2]:


def BFS(graph, vertex):
    queue = [vertex]
    visited_list = [vertex]
    while len(queue) != 0:
        vertex = queue.pop(0)
        print(vertex)
        neighbors = graph[vertex]
        for node in neighbors:
            if node in visited_list:
                continue
            else:
                visited_list.append(node)
                queue.append(node)
def BFS_A(A, idx_v):
    queue = [idx_v]
    visited_list = [idx_v]
    while len(queue) != 0:
        idx_v = queue.pop(0)
        print(idx_v)
        neighbors = A[idx_v]
        for idx_h, item in enumerate(neighbors):
            if item == 1:
                if idx_h in visited_list:
                    continue
                else:
                    visited_list.append(idx_h)
                    queue.append(idx_h)
BFS(graph, 'F')
BFS_A(A,5)


# In[3]:


graph = {
    'A':['B','C'],
    'B':['A','C','D'],
    'C':['A','B','D','E'],
    'D':['B','C','E','F'],
    'E':['C','D'],
    'F':['D']
}
def DFS(graph, vertex):
    path = []
    stack = [vertex]
    visited_list = set()
    while len(stack) != 0:
        path.append(vertex)
        visited_list.add(vertex)
        neighbors = graph[vertex]
        unvisited_neighbors = []
        for neighbor in neighbors:
            if neighbor not in visited_list:
                unvisited_neighbors.append(neighbor)
        #print(unvisited_neighbors,visited_list)
        if len(unvisited_neighbors) == 0:
            stack.pop(-1)
            if len(stack) != 0:
                vertex = stack[-1]
        else:
            vertex = unvisited_neighbors[0]
            stack.append(vertex)
            #visited_list.add(vertex)
    return path
path = DFS(graph,'F')
print(path)
A, keys = Graph2AdjMatrix(graph)
print(A, keys)


# In[4]:


def DFS_A(A, idx_v):
    vertex = idx_v
    visited_list = set()
    stack = [vertex]
    while len(stack) != 0:
        print(vertex)
        visited_list.add(vertex)
        unvisited_neighbors = []
        for idx, item in enumerate(A[vertex]):
            if item == 1 and idx not in visited_list:
                unvisited_neighbors.append(idx)
        #print(unvisited_neighbors)
        if len(unvisited_neighbors) == 0:
            stack.pop(-1)
            if len(stack) != 0:
                vertex = stack[-1]
        else:
            vertex = unvisited_neighbors[0]
            stack.append(vertex)
            #visited_list.add(vertex)
DFS_A(A,0)

