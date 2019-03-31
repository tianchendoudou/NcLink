import numpy as np
import queue as Q
S_min = [0,1,2,3,4,5]
data = [[1,1,1],[2,2,2],[4,4,4],[2,3,4],[4,3,2]]
L = {0: [0,1,2],1:[3,4]}
#
#
import heapq
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._queue)[-1]
q = PriorityQueue()
# class Skill(object):
#     def __init__(self, priority, description):
#         self.priority = priority
#         self.description = description
#
#     def __lt__(self, other):
#         return self.priority < other.priority
#
#     def __str__(self):
#         # return '(' + str(self.priority) + ',\'' + self.description + '\')'
#         return self.description
#
# que = Q.PriorityQueue()
sum = []
s = [0 for i in range(len(data[0]))]
for i in L[0]:
    s = np.sum([s,data[i]],axis = 0)
a = s/len(L[0])
dis = []
for i in L[0]:
    b = data[i]
    d1=np.sqrt(np.sum(np.square(a-b)))
    dis.append(d1)
    q.push(i,d1)
print (type(q.pop()))
print(data[q.pop()])



