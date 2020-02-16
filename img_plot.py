from model import *
from trustscore import *
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mn

#============ Read Config File ============#
num = '2'
infile = cp.ConfigParser()
infile.read("config/cfg"+num+"gmm.ini")
modelpath = "models/iccad"+num+"gmm/"
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

sess = tf.Session()
#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=50)

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

#============ TrustScore Compute ===========#
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as mn

#ckpt = tf.train.get_checkpoint_state('models/initial_model/iccad'+'3'+'gmm/')
#saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
saver.restore(sess, "models/initial_model/iccad"+num+"gmm/")
test_tmp = test_data.ft_buffer
train_tmp = train_data.ft_buffer

test_feature=test_tmp.reshape(test_tmp.shape[0], test_tmp.shape[1]*test_tmp.shape[2])
test_label=test_data.label_buffer.astype(int)
train_feature=test_tmp.reshape(test_tmp.shape[0], test_tmp.shape[1]*test_tmp.shape[2])
train_label=train_data.label_buffer.astype(int)

nc = 190
pca1 = PCA(n_components=nc)
pca2 = PCA(n_components=nc)
pca1.fit(test_feature)
pca2.fit(train_feature)
test_feature = pca1.transform(test_feature)
train_feature = pca2.transform(train_feature)

pdt = sess.run(y, feed_dict={x_data: test_tmp[:,:,:fealen]})
trust_model = TrustScore(k=10, alpha=0, filtering="density")
trust_model.fit(train_feature, train_label)
trust_score = trust_model.get_score(test_feature, pdt)

mean = np.mean(test_feature, axis=0)
sigma= np.cov(test_feature.T)
post_prob = mn.pdf(test_feature, mean=mean, cov=sigma)


data_num = {'2':700, '4':1500, '3':4700}
num_per_bin = {'2':28, '4':60, '3':188}

ts = trust_score[0:data_num[num]]
pb = post_prob[0:data_num[num]]
p0    = sess.run(predict[:,0], feed_dict={x_data: test_tmp[:,:,:fealen]})
p1    = sess.run(predict[:,1], feed_dict={x_data: test_tmp[:,:,:fealen]})

psum  = np.zeros(data_num[num])
for i in range(data_num[num]):
    if p0[i] > p1[i]:
        psum[i] = p0[i]
    else:
        psum[i] = p1[i]

#psum = p0
acc_ts = np.zeros(25)
acc_p  = np.zeros(25)
acc_pb = np.zeros(25)
ts_p   = np.zeros(25)
for i in range(25):
    idx = ts.argsort()[num_per_bin[num]*i:num_per_bin[num]*(i+1)]
    acc_ts[i] = sess.run(accu, feed_dict={x_data: test_tmp[idx][:,:,:fealen], y_gt: processlabel(test_label[idx])})
    ts_p[i] = sum(p1[idx]) / num_per_bin[num]
    idx = psum.argsort()[num_per_bin[num]*i:num_per_bin[num]*(i+1)]
    acc_p[i]  = sess.run(accu, feed_dict={x_data: test_tmp[idx][:,:,:fealen], y_gt: processlabel(test_label[idx])})
    idx = pb.argsort()[num_per_bin[num]*i:num_per_bin[num]*(i+1)]
    acc_pb[i] = sess.run(accu, feed_dict={x_data: test_tmp[idx][:,:,:fealen], y_gt: processlabel(test_label[idx])})
    
acc = sess.run(accu, feed_dict={x_data: test_tmp[:,:,:fealen], y_gt: processlabel(test_label)})

x = np.linspace(0,100,25)
plt.figure()
plt.plot(x, acc_ts, 'b-', label='Trust Score')
plt.plot(x, acc_p, 'r-', label='Prob')
plt.plot(x, acc_pb, 'y-', label='Post Prob')
plt.axvline(x=100*(1-acc), linestyle="dotted", color="black")
plt.legend(loc='lower right', prop={'size': 11})
plt.xlabel('Percentile Level')
plt.ylabel('Precision')
plt.title('Detect trustworthy')

plt.figure()
plt.plot(x, ts_p, 'b-', label='Prob1')
plt.legend(loc='upper right', prop={'size': 11})
plt.xlabel('Percentile Level')
plt.ylabel('Probability1')
#plt.title('Detect trustworthy')

plt.figure()
bins = np.linspace(0.5,2.5,10)
plt.hist(ts.astype(float), bins, histtype='bar')
plt.xlabel('Trust Score')
plt.ylabel('Quantitiy')
plt.title('Trust Score Distribution')

plt.show()