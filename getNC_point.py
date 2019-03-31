import heapq
import numpy as np
import queue as Q
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._queue)[-1]
def getNcpoint(Cluster,data):
    q = PriorityQueue()
    sum = []
    s = [0 for i in range(len(data[0]))]
    for i in Cluster:
        s = np.sum([s,data[i]],axis = 0)
    a = s/len(Cluster)
    dis = []
    for i in Cluster:
        b = data[i]
        distance = np.sqrt(np.sum(np.square(a-b)))
        dis.append(distance)
        q.push(i,distance)
    return q.pop()