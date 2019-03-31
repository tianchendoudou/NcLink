import math
class Knn:
    """docstring for Knn"""

    def __init__(self):
        self.data = []
        self.dic = {}

    # ji suan xiang lin liang ge yang ben de ou shi ju li bing yi zi dian xing shi cun chu
    def fit(self, data):
        self.data = data
        #self.real_indices = range(len(data))
        self.real_indices = list(range(len(data)))
        for i in range(len(data)):
            self.dic[(i, i)] = 0.
            for j in range(i):
                self.dic[(i, j)] = math.sqrt(math.fsum(((a - b) ** 2 for a, b in zip(self.data[i], self.data[j]))))
                self.dic[(j, i)] = self.dic[(i, j)]

    def fit_subset(self, indices):
        self.real_indices = indices

    def get_dis(self, a, b):

        return self.dic[(a, b)]

    def kneighbors(self, instance_index, n_neighbors, return_distance=False):
        result = []
        for i in self.real_indices:
            distance = self.dic[(instance_index, i)] #mei yi ge yang ben yu dang qian shu ru yang ben de ou shi ju li
            result.append((distance, i)) #yuan zu ji lu ju li he dui ying ji suan de yang ben dian
        result = sorted(result)[:n_neighbors] #pai xu ju li zui jin de yang ben dian qu k1 ge

        if return_distance:
            return ([i[1] for i in result], [i[0] for i in result])
        else:
            return [i[1] for i in result] # fan hui yang ben zhen shi suo yin