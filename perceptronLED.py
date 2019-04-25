#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# 改为python 3.7 下可运行
# modify by xmxoxo for python3.7


import numpy as np
from functools import reduce


#感知器类
class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0
    def bias(self,b):
        self.bias = b

    def weights (self,w):
        self.weights = w

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s bias\t:%f\n' % (list(self.weights), self.bias)
    
    #返回感知器的所有参数
    def getparm(self):
        return (list(self.weights), self.bias)
    
    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        # python3.7的 map reduce函数使用方法不同

        '''
        # 如果zip的话 这样也可以
        return self.activator(
            reduce(lambda a, b: a + b,
                    map(lambda x: x[0] * x[1],
                        zip(input_vec, self.weights)
                       )
                  ,0.0) + self.bias)
        '''

        return self.activator(
            reduce(lambda a, b: a + b,
                    map(lambda x,w: x * w ,
                        input_vec, self.weights
                       )
                  ,0.0) + self.bias)


    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
            #输出训练过程
            #print(self.__str__())

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        # python3.7 中需要指定返回类型为list
        delta = label - output
        self.weights = list(map(
            lambda x,w: w + rate * delta * x,
            input_vec, self.weights))
        # 更新bias
        self.bias += rate * delta


def f(x):
    '''
    定义激活函数f
    '''
    return 1 if x > 0 else 0


#构造训练数据集 LED 7段数码
def get_training_dataset(index=0):
    if not (0<=index<=6):
        index = 0
    # 输入向量列表
    input_vecs = [
            [0,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,0,1,1],
            [0,1,0,0],
            [0,1,0,1],
            [0,1,1,0],
            [0,1,1,1],
            [1,0,0,0],
            [1,0,0,1],
            [1,0,1,0],
            [1,0,1,1],
            [1,1,0,0],
            [1,1,0,1],
            [1,1,1,0],
            [1,1,1,1],
        ]
    lst_labels = [
            [1,0,1,1,0,1,1,1,1,1,1,0,1,0,1,1],
            [1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,0],
            [1,1,0,0,1,1,1,1,1,1,1,1,0,1,0,0],
            [1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,0],
            [1,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1],
            [1,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1],
            [0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1],
        ]

    labels = lst_labels[index]

    return input_vecs, labels    


def train_perceptron(index=0):
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为4,激活函数为f
    p = Perceptron(4, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset(index)
    p.train(input_vecs, labels, 15, 0.1)
    #返回训练好的感知器
    return p

#测试真值表
def test_table (obj,index):
    pass
    print('测试结果'.center(30,'-'))
    input_vecs, labels = get_training_dataset(index)
    predict = [obj.predict(item) for item in input_vecs]
    acc = list(map(lambda x,y: int(x==y), labels,predict))
    print(labels)
    print(predict)
    print(acc)
    print('准确率: %2.2f' % (sum(acc)/len(acc) ) )

    #for item in input_vecs:
    #    print ('%s = %d' % ( str(item) ,obj.predict(item)))

##训练7个感知器
def transLED ():
    pass
    #使用一个数组来保存训练结果
    lstModel = []
    strSeg = "abcdefg"
    for i in range(len(strSeg)):        
        print ('--------正在训练%s感知器--------' % strSeg[i])
        # 训练感知器
        perception = train_perceptron(i)
        # 打印训练获得的权重
        print (str(perception))
        #保存到模型中
        lstModel.append(perception.getparm())
        # 测试
        test_table(perception,i)
        break #先训练一个
        return 0
        
    #模型训练完成，输出下结果：
    print("--------模型训练完成,各感知器参数如下：--------")
    for i in range(len(lstModel)):
        print("%s 感知器参数: %s" % (strSeg[i], str(lstModel[i]) ))


##将7bit转换成LED显示
def showLED (dat):
    pass
    if type(dat)==str:
        lstDat = list(map(lambda x:int(x),list(dat)))
    else:
        lstDat = dat
    ret = []
    for i in range(len(lstDat)):
        if i in [0,3,6]:
            ret.append( '-' if lstDat[i] else ' ') 
        else:
            ret.append( '|' if lstDat[i] else ' ') 
    #print(ret)
    strTxt = ' %s \n%s %s\n %s \n%s %s\n %s ' % (ret[0],ret[5],ret[1],ret[6],ret[4],ret[2],ret[3])
    return strTxt

#测试所有LED字符
def testAllchar ():
    print('测试LED字符输出:')
    sTxt = '0123456789AbCdEF'
    code = ["1111110","0110000","1101101","1101101", \
            "0110011","1011011","1011111","1110000", \
            "1111111","1111011","1110111","0011111", \
            "1001110","0111101","1001111","1000111"]
    for i in range(16):
        print('-'*30)
        print(sTxt[i],code[i])
        print(showLED(code[i]))
    

if __name__ == '__main__': 
    testAllchar()

    #transLED()

#python perceptronLED.py>log.txt
