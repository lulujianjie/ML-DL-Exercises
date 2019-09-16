
# coding: utf-8

# In[94]:


graph = {
    'A':{'B':5, 'C':1},
    'B':{'A':5, 'C':2, 'D':1},
    'C':{'A':1, 'B':2, 'D':4, 'E':8},
    'D':{'B':1, 'C':4, 'E':3, 'F':6},
    'E':{'C':8, 'D':3},
    'F':{'D':6}
}


# In[104]:


import heapq
parent_dict = {}
start_node = 'A'
visited = set()
pqueue = []
heapq.heappush(pqueue, (0, start_node))
parent_node = 'N/A'
while(len(pqueue) > 0):
    pair = heapq.heappop(pqueue)
    print(pair)
    distance = pair[0]
    current_node = pair[1]
    if current_node not in visited:
        parent_dict[current_node] = (distance, parent_node)
    #print(current_node)
    visited.add(current_node)
    neighbors = graph[current_node].keys()
    for neighbor in neighbors:
        if neighbor not in visited:
            heapq.heappush(pqueue, (graph[current_node][neighbor], neighbor))
            parent_node = current_node


# In[105]:


parent_dict

