import os
import string
import numpy as np
from itertools import islice
import random
import csv
import pandas as pd 
from scipy.fftpack import dct

from time import time
import json
from scipy.misc import *
import skimage.measure as skm
import math
import multiprocessing as mtp
import cvxopt as cvx


def debug():
    print("=============================================================================================")



def binsearch(x):
    interval= list(range(500))
    return bisect.bisect_left(interval, x*500)    


def quantization(size, val=1):
    return np.empty(size*size, dtype=int).reshape(size,size).fill(val)

def rescale(img):
    return (img/255)

#calculate 2D DCT of a matrix
def dct2(img):
    return dct(dct(img.T, norm = 'ortho').T, norm = 'ortho')

def softmax(y, type=1):
    sum=0.0
    for i in range(0, len(y)):
        sum+=math.exp(y[i])
    return math.exp(y[type])/sum

def subfeature(imgraw, quanti, fealen):

    if not ((len(imgraw) == len(quanti)) and (len(imgraw[:,0]) == len(quanti[:,0]))):
        print ('ERROR: Image block must have the same size as Quantization matrix.')
        print ('Abort.')
        quit()
    if fealen > len(imgraw)*len(imgraw[:,0]):
        print ('ERROR: Feature vector length exceeds block size.')
        print ('Abort.')
        quit()

    img =dct2(imgraw)
    size=fealen
    idx =0
    scaled=np.divide(img, quanti)
    feature=np.zeros(fealen, dtype=np.int)
    for i in range(0, size):
        if idx>=size:
            break
        elif i==0:
            feature[0]=scaled[0,0]
            idx=idx+1
        elif i%2==1:
            for j in range(0, i+1):
                if idx<size:
                    feature[idx]=scaled[j, i-j]
                    idx=idx+1
                else:
                    break
        elif i%2==0:
            for j in range(0, i+1):
                if idx<size:
                    feature[idx]=scaled[i-j, j]
                    idx=idx+1
                else:
                    break

    return feature


def cutblock(img, block_size, block_dim):
    blockarray=[]
    for i in range(0, block_dim):
        blockarray.append([])
        for j in range(0, block_dim):
            blockarray[i].append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])

    return np.asarray(blockarray)


def feature(img, block_size, block_dim, quanti, fealen):
    img=rescale(img)
    feaarray = np.empty(fealen*block_dim*block_dim).reshape(fealen, block_dim, block_dim)
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            featemp=subfeature(blocked[i,j], quanti, fealen)
            feaarray[:,i,j]=featemp
    return feaarray

def kl(x, y):
    size=len(x)
    kld=0

    for i in range(size):
        kld+=(x[i]-y[i])*(np.exp(x[i])-np.exp(y[i]))/(np.exp(x[i])+np.exp(y[i]))

    return kld
#####################################
#Building div mtx

def unwrap_self_f(arg, **kwarg):
    return distance_mtx.distance_l2(*arg, **kwarg)
class distance_mtx:
    def __init__(self, fea):
        self.x=fea
        self.dim=len(fea)
        self.mtx=np.zeros((self.dim, self.dim))
        self.idxs=-1
    def create_idxs(self):
        num=self.dim
        base=np.array(range(num))
        a=np.repeat(base, num)
        b=np.tile(base, num)
        c=np.concatenate(([a],[b]), axis=0).T
  
        self.idxs=c
############################################



def entropy(x):
    a=np.exp(x[0])/(np.exp(x[0])+np.exp(x[1]))
    b=np.exp(x[1])/(np.exp(x[0])+np.exp(x[1]))
    return -aloga(a)-aloga(b)

def aloga(x):
    if x==0:
        return 0
    else:
        return x*math.log(x)



##############################################
#Solving QP with CVXOPT
#
#

class qp_cvx:
    def __init__(self, mtx, hs_score, k, alpha, solver=0):

        self.q=cvx.matrix((-hs_score*alpha).tolist())
        self.dim=len(hs_score)
        self.G=cvx.matrix(np.concatenate((-np.identity(self.dim), -np.identity(self.dim))))
        self.h=cvx.matrix(np.concatenate((np.zeros(self.dim), np.ones(self.dim))))
        self.A=cvx.matrix(np.expand_dims(np.ones(self.dim), axis=0))
        self.b=cvx.matrix(np.array([[k]]).astype(float))
        self.x=np.zeros(self.dim)
        self.obj=0
        self.opt=solver
        
        if self.opt==0:
            self.P=cvx.matrix(-mtx)
        if self.opt==1:
            self.P=cvx.matrix(mtx)
    def solve(self):
        solvers=cvx.solvers
        solvers.options['show_progress'] = False
        if self.opt==0:
            qp=solvers.qp(self.P,self.q,self.G,self.h,self.A,self.b, kktsolver='ldl', options={'kktreg':1e-9})
        if self.opt==1:
            qp=solvers.qp(self.P,self.q,self.G,self.h,self.A,self.b)
        self.x=np.array(qp['x']).flatten()
        self.obj=qp['primal objective']
































"""
def forwardAL(net, is_training = True, reuse=False):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      #activation_fn=tf.contrib.keras.layers.LeakyReLU,
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                      biases_initializer=tf.constant_initializer(0)):
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='d_conv1')
        net = slim.conv2d(net, 64, [3, 3], stride=2, scope='d_pool1') #100
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='d_conv2')
        net = slim.conv2d(net, 128, [3, 3], stride=2, scope='d_pool2')#50
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='d_conv3')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='d_pool3')#25
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='d_conv4')
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='d_pool4')#12
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='d_conv5')
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='d_pool5')#6
        net = slim.flatten(net)
        net = slim.fully_connected(net, 2048, scope='d_fc6')
        net = slim.dropout(net, 0.5, scope='d_dropout6', is_training=is_training)
        net = slim.fully_connected(net, 512, scope='d_fc7')
        net = slim.dropout(net, 0.5, scope='d_dropout7', is_training=is_training)
        net = slim.fully_connected(net, 2, activation_fn=None, scope='d_fc8')
    return net
"""

def stochasticInputProducer(lists, batchSize=0):

    dataLength = len(lists)
    randSampling = random.sample(range(0, dataLength), batchSize)
    if batchSize==0:
        batchSize=dataLength
        batchList = lists
        label=np.zeros([dataLength, 2], dtype=int)
    else:
        batchList = lists[randSampling]
        label=np.zeros([batchSize, 2], dtype=int)
    #print batchList
    for i in range(batchSize):
        tmp=imread(batchList[i].split()[0], mode='L')
        tmpLabel=int(batchList[i].split()[1])
    #print tmpLabel
        label[i, tmpLabel]=1

        tmp=np.expand_dims(tmp, axis=0)
        if i==0:
            batchData=tmp
        else:
            batchData=np.concatenate((batchData,tmp), axis=0)
    batchData=np.expand_dims(batchData, axis=3)
    #print label, batchData.shape2
    return label, batchData

def batchInputProducer(imgLoc, batchSize=0, pointer=0):
    with open(imgLoc) as f:
        lists=np.array(f.readlines())

    dataLength = len(lists)
    start = pointer
    if pointer + batchSize >= dataLength:
        end = dataLength
    else:
        end = pointer + batchSize



    batchList = lists[start:end]
    batchSizeNew = len(batchList)
    label=np.ones([batchSizeNew, 2], dtype=int)
    #print batchList
    for i in range(batchSizeNew):
        tmp=imread(batchList[i].split()[0], mode='L')/255
        tmpLabel=int(batchList[i].split()[1])
    #print tmpLabel
        label[i, tmpLabel]=0

        tmp=np.expand_dims(tmp, axis=0)
        if i==0:
            batchData=tmp
        else:
            batchData=np.concatenate((batchData,tmp), axis=0)
    batchData=np.expand_dims(batchData, axis=3)
    if pointer + batchSize >= dataLength:
        pointer = 0
    else:
        point = end
    #print label, batchData.shape
    return label, batchData, pointer


'''
    readcsv: Read feature tensors from csv data packet
    args:
        target: the directory that stores the csv files
        fealen: the length of feature tensor, related to to discarded DCT coefficients
    returns: (1) numpy array of feature tensors with shape: N x H x W x C
             (2) numpy array of labels with shape: N x 1 
'''
def readcsv(target, fealen=32):
    #read label
    path  = target + '/label.csv'
    label = np.genfromtxt(path, delimiter=',')
    #read feature
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in range(0, len(filenames)-1):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature.append(featemp)          
    return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen], label

def writecsv(target, data, label, fealen):
    #flatten data
    data=data.reshape(len(data), fealen, len(data[0,0,0,:])*len(data[0,0,:,0]))
    for i in range(0, fealen):
        if i == 0:
            path = target + '/dc.csv'
            np.savetxt(path, 
                data[:,i,:],
                fmt='%d',
                delimiter=',',          # column delimiter
                newline='\n',           # new line character
                comments='#',          # character to use for comments
                )
        else:
            path = target + '/ac' +str(i) + '.csv'
            np.savetxt(path,
                data[:,i,:],
                fmt='%d',
                delimiter=',',
                newline='\n',
                comments='#')
    path = target+'/label.csv'
    np.savetxt(path,
        label,
        fmt='%d',
        delimiter=',',
        newline='\n',
        comments='#')
def writecsv3(target, data, label, fealen):
    #flatten data
    #data=data.reshape(len(data), fealen, len(data[0,0,0,:])*len(data[0,0,:,0]))
    for i in range(0, fealen):
        if i == 0:
            path = target + '/dc.csv'
            np.savetxt(path, 
                data[:,:,i],
                fmt='%d',
                delimiter=',',          # column delimiter
                newline='\n',           # new line character
                comments='#',          # character to use for comments
                )
        else:
            path = target + '/ac' +str(i) + '.csv'
            np.savetxt(path,
                data[:,:,i],
                fmt='%d',
                delimiter=',',
                newline='\n',
                comments='#')
    path = target+'/label.csv'
    np.savetxt(path,
        label,
        fmt='%d',
        delimiter=',',
        newline='\n',
        comments='#')
'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel
'''
    loss_to_bias: calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
'''
def loss_to_bias(loss, threshold=0.3):
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1+np.exp(43*loss))
    return bias
def distance(s):
    return np.linalg.norm(feature[s[0]]-feature[s[1]])


'''
    data: a class to handle the training and testing data, implement minibatch fetch
    args: 
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args: 
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

'''
class data:
    def __init__(self, fea, lab, preload=False):
        self.ptr_n=0
        self.ptr_h=0
        self.ptr=0
        self.dat=fea
        self.label=lab
        self.preload=preload
        
        with open(lab) as f:
            self.maxlen=sum(1 for _ in f)
        if preload:
            print("loading data into the main memory...")
            self.ft_buffer, self.label_buffer=readcsv(self.dat)
            self.fealen=self.ft_buffer.shape[-1]
    def stat(self):
        if not self.preload:
            print("preload data required, abort!")
        else:
            total=self.maxlen
            hs=sum(self.label_buffer)
            nhs=total - hs
            return hs, nhs
    def reset(self):
        self.ptr=0
        self.ptr_h=0
        self.ptr_n=0
        self.maxlen=len(self.label_buffer)
    def to_csv(self, target):
        for i in range(0, self.fealen):
            if i == 0:
                path = os.path.join(target, 'dc.csv')
                df= pd.DataFrame(data=np.array(self.ft_buffer[:,:,i]))
                df.to_csv(path, header=False, index=False)
            else:
                path = os.path.join(target, 'ac' +str(i) + '.csv') 
                df= pd.DataFrame(data=np.array(self.ft_buffer[:,:,i]))
                df.to_csv(path, header=False, index=False)
        path = os.path.join(target, 'label.csv')
        np.savetxt(path,
            self.label_buffer,
            fmt='%d',
            delimiter=',',
            newline='\n',
            comments='#')        
    def nextinstance(self):
        temp_fea=[]
        label=None
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))        
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])
    
    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        #label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''
    def nextbatch_beta(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length        #感觉会进入下一个循环 不断有数据被选出来
            return ptr
        #with open(self.label) as l:
        labelist=self.label_buffer
        labexn = np.where(labelist==0)[0]
        labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size

        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = min(batch//2, n_length, h_length)
            #print num
            if 1==2:
                print('ERROR:Batch size exceeds data size')
                print('Abort.')
                quit()
            else:
                if self.ptr_n+num <n_length:
                    idxn = labexn[self.ptr_n:self.ptr_n+num]
                elif self.ptr_n+num >=n_length:
                    idxn = np.concatenate((labexn[self.ptr_n:n_length], labexn[0:self.ptr_n+num-n_length]))
                self.ptr_n = update_ptr(self.ptr_n, num, n_length)
                if self.ptr_h+num <h_length:
                    idxh = labexh[self.ptr_h:self.ptr_h+num]
                elif self.ptr_h+num >=h_length:
                    idxh = np.concatenate((labexh[self.ptr_h:h_length], labexh[0:self.ptr_h+num-h_length]))
                self.ptr_h = update_ptr(self.ptr_h, num, h_length)
                #print self.ptr_n, self.ptr_h
                label = np.concatenate((np.zeros(num), np.ones(num)))
                #label = processlabel(label,2, delta1, delta2)
                ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
                ft_batch_nhs = self.ft_buffer[idxn]
                label_nhs = np.zeros(num)
        return ft_batch[:,:,:channel], label, ft_batch_nhs[:,:,:channel], label_nhs
    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch[:,:,:channel], label
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            #processing labels
            with open(self.label) as l:
                temp_label=np.asarray(list(l)[self.ptr:self.ptr+batch])
                label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:
            
            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
            self.ptr=self.ptr+batch-self.maxlen
        #print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label

'''
    forward: define the neural network architecute
    args:
        input: feature tensor batch with size B x H x W x C
        is_training: whether the forward process is training, affect dropout layer
        reuse: undetermined
        scope: undetermined
    return: prediction socre(s) of input batch
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
def forward(input, is_training=True, reuse=False, scope='model'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=None, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0)):
            net = slim.conv2d(input, 16, [3, 3], scope='conv1_1')
            net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn1_1')
            net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
            net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn1_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
            #net = slim.conv2d(net, 16, [3, 3], stride=2, scope='conv1_3')
            #net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn1_3')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
            net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn2_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
            net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn2_2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            #net = slim.conv2d(net, 32, [3, 3], stride=2, scope='conv2_3')
            #net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn2_3')
            #net = slim.conv2d(net, 32, [3, 3], scope='conv3_1')
            #net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn3_1')
            #net = slim.conv2d(net, 32, [3, 3], scope='conv3_2')
            #net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn3_2')
            #net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
            #net = slim.conv2d(net, 32, [3, 3], stride=2, scope='conv3_3')
            #net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bn3_3')
            net = slim.flatten(net)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False)
            net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
            fea=net
            net = slim.batch_norm(net, center=True, scale=True, activation_fn=tf.nn.relu, is_training=is_training, updates_collections=None, reuse=reuse, scope='bnfc1')
            
            net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')

            predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
    return predict, fea
