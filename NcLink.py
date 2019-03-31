from source import *
from knn import *
import numpy as np
import queue as Q
from getNC_point import *

que = Q.PriorityQueue()
k = Knn()
X,S_min = source()
T_h = 60 # T_h 是阈值，C_p作为恒定参数控制类簇规模
k.fit(X)

def Euclidean_Distance(a,b):
    d1=np.sqrt(np.sum(np.square(a-b)))
    return d1

def Ncpoint(cluster,X):
    index = getNcpoint(cluster,X)
    return index

def clus_dis(A, B,X,K):
    if len(A) == 1 and len(B) == 1:
        NC_pointA = A[0]
        NC_pointB = B[0]
        # print(NC_pointA)
        # print(NC_pointB)
        distance = K.get_dis(NC_pointA, NC_pointB)
    else:
        NC_pointA = Ncpoint(A,X)
        NC_pointB = Ncpoint(B,X)
        # print(NC_pointA)
        # print(NC_pointB)
        distance = K.get_dis(NC_pointA, NC_pointB)
    return distance


L = {index: [i] for index, i in enumerate(S_min)}
clusters_number = list(range(len(S_min)))
# clusters_number = range(len(S_min))
dis_table = [[0 for i in clusters_number] for j in clusters_number]
for i in clusters_number:
    for j in clusters_number:
        dis_table[i][j] = clus_dis(L[i], L[j], X,k)
MAX = max(max(j) for j in dis_table)
for i in clusters_number:
    dis_table[i][i] = MAX
for i in S_min:
    MIN = min(min(j) for j in dis_table)
    if MIN > T_h:
        break
    for j in clusters_number:
        if MIN in dis_table[j]:
            b = dis_table[j].index(MIN)
            a = j
            break
    L[a].extend(L[b])

    del L[b]
    clusters_number.remove(b)
    for j in clusters_number:
        tmp = clus_dis(L[a], L[j],X,k)
        dis_table[a][j] = tmp
        dis_table[j][a] = tmp
    dis_table[a][a] = MAX
    for j in clusters_number:
        dis_table[b][j] = MAX
        dis_table[j][b] = MAX

which_cluster = {}
for i, clu in L.items():
    for j in clu:
        which_cluster[j] = i

print(len(L))