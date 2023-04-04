#!/usr/bin/env python
# coding: utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt
import math
import pickle


class NN:
    def __init__(self, traindata, l, regularization_factor, hiddenlayersize, cos_st):
        self.datanum = len(traindata)  
        self.traindata = traindata  
        self.lrate = l  
        self.cos_st = cos_st  
        self.startlrate = l
        self.reg_factor = regularization_factor  
        self.hiddensize = hiddenlayersize  
        self.batchsize = 100
        self.loss = []
        self.trainacc = []

        # 初始化网络(两层) 0-9 10个手写数字类别
        self.W1 = (np.random.rand(28 * 28, self.hiddensize) - 0.5) * 2/ 28
        self.b1 = np.zeros((1, self.hiddensize))
        self.W2 = (np.random.rand(self.hiddensize, 10) - 0.5) * 2 / np.sqrt(self.hiddensize)
        self.b2 = np.zeros((1, 10))

        self.allW1 = [self.W1]
        self.allW2 = [self.W2]
        self.allb1 = [self.b1]
        self.allb2 = [self.b2]
    
    def train(self, epoch_num):
        iternum = self.datanum//self.batchsize
        for epoch in range(epoch_num):
            np.random.shuffle(self.traindata)
            for i in range(iternum):
                self.lrate = self.startlrate
                images = self.traindata[i * self.batchsize: (i+1) * self.batchsize, :-1]
                labels = self.traindata[i * self.batchsize: (i+1) * self.batchsize, -1:]
                self.trainacc.append(self.predict(images, labels))
                self.update(images, labels)
            
    
    def update(self, data, labels):
        hiddenlayer_output = np.maximum(np.matmul(data, self.W1) + self.b1, 0)
        outlayer = np.maximum(np.matmul(hiddenlayer_output, self.W2) + self.b2, 0)

        scores = np.exp(outlayer)  
        scores_sum = np.sum(scores, axis=1, keepdims=True)  
        
        temp = np.empty((self.batchsize, 1))
        for i in range(self.batchsize):
            temp[i] = scores[i][int(labels[i])]/scores_sum[i]
        crossentropy = - np.log(temp)

        # 损失函数 = 交叉熵损失项 + L2正则化项
        loss = np.mean(crossentropy, axis=0)[0] + 0.5 * self.reg_factor * (np.sum( self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        self.loss.append(loss) 

        # 反向传播更新参数
        res = scores / scores_sum  
        for i in range(self.batchsize):
            res[i][int(labels[i])] -= 1
        res  /= self.batchsize
        
        dW2 = np.matmul( hiddenlayer_output.T, res)  
        db2 = np.sum(res, axis=0, keepdims=True) 

        # 由递推公式得到第一层的残差
        dh1 = np.dot( res, self.W2.T) 
        dh1[hiddenlayer_output<=0] = 0 

        dW1 = np.dot( data.T, dh1)
        db1 = np.sum( dh1, axis=0, keepdims=True)

        # L2正则化求导项
        dW2 += self.reg_factor * self.W2
        dW1 += self.reg_factor * self.W1

        # 更新参数
        self.W2 += -self.lrate * dW2
        self.W1 += -self.lrate * dW1
        self.b2 += -self.lrate * db2
        self.b1 += -self.lrate * db1

        self.allW1.append(self.W1)
        self.allW2.append(self.W2)
        self.allb1.append(self.b1)
        self.allb2.append(self.b2)
        
        return
    
    def visualization(self):
        plt.plot(self.loss)
        plt.xlabel('batches')
        plt.ylabel('train loss')
        plt.title('lr:' + str(self.startlrate) + ' reg_factor:' + str(self.reg_factor) + ' hiddensize:' + str(self.hiddensize))
        plt.show()

    def predict(self, testdata, testlabel):
        hiddenlayer_output = np.maximum(np.matmul(testdata, self.W1) + self.b1, 0)
        outlayer = np.maximum(np.matmul(hiddenlayer_output, self.W2) + self.b2, 0)
        prediction = np.argmax(outlayer, axis=1).reshape((len(testdata),1))
        accuracy = np.mean(prediction == testlabel)
        return accuracy
    
    def getProcess(self):
        return self.allW1, self.allW2, self.allb1, self.allb2, self.loss, self.trainacc
    
    def savemodel(self):
        paras = {}
        paras['W1'] = self.W1
        paras['W2'] = self.W2
        paras['b1'] = self.b1
        paras['b2'] = self.b2
        with open('bestpara.pkl', 'wb') as f:
            pickle.dump(paras, f)
    

        


def findbest(traindata, testdata, testlabel):
    acclist = {}
    for lr in [0.005, 0.01, 0.05]:
        for regul in [0.001, 0.005, 0.01]:
            for hiddensize in [100, 200, 300]:
                a = NN(traindata, lr, regul, hiddensize, False)
                a.train(5)
                acclist[str(lr) + '/' + str(regul) + '/' + str(hiddensize)] = a.predict(testdata, testlabel)
    return max(list(acclist.values())), list(acclist.keys())[list(acclist.values()).index(max(list(acclist.values())))], acclist

def finalmodel(lr, regu, hsize, traindata):
    a = NN(traindata, lr, regu, hsize, False)
    a.train(5)
    a.savemodel()
    processpara = {}
    processpara['allW1'] = a.getProcess()[0]
    processpara['allW2'] = a.getProcess()[1]
    processpara['allb1'] = a.getProcess()[2]
    processpara['allb2'] = a.getProcess()[3]
    processpara['processloss'] = a.getProcess()[4]
    processpara['trainacc'] = a.getProcess()[5]

    with open('processparameters.pkl', 'wb') as f:
        pickle.dump(processpara, f)
    
    # 可视化最后选取模型训练的loss/acc
    x = processpara['processloss']
    plt.plot(x, label='loss')
    z = processpara['trainacc']
    plt.plot(z, label='acc')
    plt.xlabel('iteration num')
    plt.ylabel('acc/loss ')
    plt.title('training accuary and loss')
    plt.legend()
    plt.show()




