#@Time      :2018/9/14 14:27
#@Author    :zhounan
# @FileName: mlknn.py

import numpy as np

# find k neighbors
def knn(train_x, t_index, k):
    data_num = train_x.shape[0]
    dis = np.zeros(data_num)
    neighbors = np.zeros(k)

    for i in range(data_num):
        dis[i] = ((train_x[i] - train_x[t_index]) ** 2).sum()

    for i in range(k):

        temp = float('inf')
        temp_j = 0
        for j in range(data_num):
            if (j != t_index) and (dis[j] < temp):
                temp = dis[j]
                temp_j = j
        dis[temp_j] = float('inf')
        neighbors[i] = temp_j

    return neighbors


def knn_test(train_x, t, k):
    data_num = train_x.shape[0]
    dis = np.zeros(data_num)
    neighbors = np.zeros(k)

    for i in range(data_num):
        dis[i] = ((train_x[i] - t) ** 2).sum()

    for i in range(k):

        temp = float('inf')
        temp_j = 0
        for j in range(data_num):
            if dis[j] < temp:
                temp = dis[j]
                temp_j = j
        dis[temp_j] = float('inf')
        neighbors[i] = temp_j

    return neighbors

def evaluation():
    test_y = np.load('dataset/test_y.npy')
    predict = np.load('parameter_data/predict.npy')
    test_y = test_y.astype(np.int)

    hamming_loss = HammingLoss(test_y, predict)
    print('hamming_loss = ', hamming_loss)

def HammingLoss(test_y, predict):
    label_num = test_y.shape[1]
    test_data_num = test_y.shape[0]
    hamming_loss = 0
    temp = 0
    for i in range(test_data_num):
        temp = temp + np.sum(test_y[i] ^ predict[i])

    hamming_loss = temp / label_num / test_data_num

    return hamming_loss


class MLKNN(object):
    s = 1
    k = 10
    label_num = 0
    train_data_num = 0
    train_x = np.array([])
    train_y = np.array([])

    Ph1 = np.array([])
    Ph0 = np.array([])
    Peh1 = np.array([])
    Peh0 = np.array([])

    def __init__(self, train_x, train_y, k ,s):
        self.k = k
        self.s = s
        self.train_x = train_x
        self.train_y = train_y
        self.label_num = train_y.shape[1]
        self.train_data_num = train_x.shape[0]
        self.Ph1 = np.zeros(self.label_num)
        self.Ph0 = np.zeros(self.label_num)
        self.Peh1 = np.zeros([self.label_num, self.k + 1])
        self.Peh0 = np.zeros([self.label_num, self.k + 1])

    def train(self):
        #computing the prior probabilities
        for i in range(self.label_num):
            cnt = 0
            for j in range(self.train_data_num):
                if self.train_y[j][i] == 1:
                    cnt = cnt + 1
            self.Ph1[i] = (self.s + cnt) / (self.s * 2 + self.train_data_num)
            self.Ph0[i] = 1 - self.Ph1[i]

        for i in range(self.label_num):

            print('training for label\n', i + 1)
            c1 = np.zeros(self.k + 1)
            c0 = np.zeros(self.k + 1)

            for j in range(self.train_data_num):
                temp = 0
                neighbors = knn(self.train_x, j, self.k)

                for k in range(self.k):
                    temp = temp + int(self.train_y[int(neighbors[k])][i])

                if self.train_y[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1

            for j in range(self.k + 1):
                self.Peh1 = (self.s + c1[j]) / (self.s * (self.k + 1) + np.sum(c1))
                self.Peh0 = (self.s + c0[j]) / (self.s * (self.k + 1) + np.sum(c0))


    def save(self):
        np.save('parameter_data/Ph1.npy', self.Ph1)
        np.save('parameter_data/Ph0.npy', self.Ph0)
        np.save('parameter_data/Peh1.npy', self.Peh1)
        np.save('parameter_data/Peh0.npy', self.Peh0)

    def load(self):
        Ph1 = np.load('parameter_data/Ph1.npy')
        Ph0 = np.load('parameter_data/Ph0.npy')
        Peh1 = np.load('parameter_data/Peh1.npy')
        Peh0 = np.load('parameter_data/Peh0.npy')

    def test(self):
        test_x = np.load('dataset/test_x.npy')
        test_y = np.load('dataset/test_y.npy')
        predict = np.zeros(test_y.shape, dtype=np.int)
        test_data_num = test_x.shape[0]

        for i in range(test_data_num):
            neighbors = knn_test(self.train_x, test_x[i], k)

            for j in range(self.label_num):
                temp = 0
                for nei in neighbors:
                    temp = temp + int(self.train_y[int(nei)][j])

                if(self.Ph1[j] * self.Peh1[j][temp] > self.Ph0[j] * self.Peh0[j][temp]):
                    predict[i][j] = 1
                else:
                    predict[i][j] = 0

        np.save('parameter_data/predict.npy', predict)

if __name__ == '__main__':
    k = 10
    s = 1
    train_x = np.load('dataset/train_x.npy')
    train_y = np.load('dataset/train_y.npy')

    mlknn = MLKNN(train_x, train_y, k, s)
    #mlknn.train()
    #mlknn.save()
    #mlknn.load()
    #mlknn.test()

    evaluation()