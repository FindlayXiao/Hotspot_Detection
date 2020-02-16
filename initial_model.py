from model import *
import tensorflow as tf
import sys
import os
import time
import cvxopt as cvx
from datetime import datetime
import multiprocessing as mtp
import configparser as cp
import copy
import pandas as pd 
import pickle
#============ Read Config File ============#
num = '4'
infile = cp.ConfigParser()
infile.read("config/cfg"+num+"gmm.ini")
modelpath = "models/initial_model/iccad"+num+"gmm/"
datapath  = "benchmarks/fts/data"+num
filename  = "H_set"+num+".data"
fealen=32
isExists=os.path.exists(modelpath)
if not isExists:
    os.makedirs(modelpath) 

maxitr =int(infile.get('al','maxitr'))
phs_co = float(infile.get('al','phs_co'))
additional_instance_num = int(infile.get('al','additional_instance_num'))
pool_size_upper = int(infile.get('al','pool_size_upper'))
additional_instance_per_iter = int(infile.get('al','additional_instance_per_iter'))

start_to_active = int(infile.get('al','start_to_active'))
delta1=float(infile.get('al','delta1'))
max_delta1=float(infile.get('al','max_delta1'))
delta2=float(infile.get('al','delta2'))
#ckpt=int(infile.get('al','ckpt'))
#delta1=0
div_channel=int(infile.get('al','div_channel'))
init_idx=datapath+'gmm'+'.csv'	#初始训练数据

#============ Import Datasets  ============#
data=data(datapath, os.path.join(datapath,'label.csv'), preload=True) #总数据
blockdim=np.sqrt(data.ft_buffer.shape[1]).astype(int)

train_data=copy.copy(data)
test_data=copy.copy(data)
test_data_for_test=copy.copy(data) #不在训练集中的data
drop_data=copy.copy(data)

#============ Define Model     ============#
x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])    #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                            #ground truth label
lr_holder=tf.placeholder(tf.float32, shape=[])									#feed in learning rate
#y_gt_c = tf.placeholder(tf.float32, shape=[None, 2])                           
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                   #reshap to NHWC

predict, fea= forward(x, is_training=False)                                     #do forward
loss   = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict)
loss   = tf.reduce_mean(loss)

y      = tf.cast(tf.argmax(predict, 1), tf.int32)								
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                                                    #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
gs     = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)       	#define global step
#lr     =  tf.train.exponential_decay(0.001, gs, decay_steps=20000, decay_rate = 0.65, staircase = True) #initial learning rate and lr decay
lr_base     = 0.0001
lr=lr_base
dr     = 1.0#learning rate decay rate
opt    = tf.train.AdamOptimizer(lr_holder, beta1=0.9)
opt    = opt.minimize(loss, global_step=gs)				#gs用于记录全局训练步骤

bs     = 32   					#training batch size
l_step = 200   					#display step
d_step = 1000 					#lr decay step
test   =1	 					#5 for iccad 12  1 for iccad16
stop   =0

additional_instance_num = min(test_data.maxlen, additional_instance_num)
sample_iters = (additional_instance_num-start_to_active)//pool_size_upper
initer= start_to_active/additional_instance_per_iter

#============ GPU Distribution ============#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.45
DEBUG=True
#DEBUG=False

#============ ---------------- ============#
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)

#============ DataSet Partition ===========#
index_test = np.genfromtxt(init_idx, delimiter=',').astype(int)
mask = np.ones(len(test_data.label_buffer), dtype=bool)
mask[index_test]=False

test_data.ft_buffer=test_data.ft_buffer[index_test]
test_data.label_buffer=test_data.label_buffer[index_test]
train_data.ft_buffer=train_data.ft_buffer[mask]
train_data.label_buffer=train_data.label_buffer[mask]

train_data.reset()
test_data.reset()  
test_data_for_test.reset()     
print (train_data.ft_buffer.shape, test_data.ft_buffer.shape)

#============ Training Initialize Model ===========#
print("Training Initialize Model......")
for step in range(maxitr*4):
    batch = train_data.nextbatch_beta(bs, fealen)   #一个batch含32个数据 且16个hs 16个nh
    batch_data = batch[0]
    batch_label= batch[1]
    batch_label_all_without_bias = processlabel(batch_label, delta1=0)

    training_loss, training_acc = \
        sess.run(loss, feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias}), sess.run(accu, feed_dict={x_data:batch_data, y_gt:batch_label_all_without_bias})
    opt.run(session=sess, feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias, lr_holder: lr})
    if (step+1) % l_step == 0:      #
        format_str = ('%s: step %d, loss = %.2f, training_accu = %f')
        print (format_str % (datetime.now(), step, training_loss, training_acc))
    if step == maxitr*4-1:
        path = modelpath
        saver.save(sess, path)
    if step % d_step == 0 and step >0:
        lr = lr * dr
lr=lr_base

sess.close()