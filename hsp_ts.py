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
import warnings 
warnings.filterwarnings('ignore')

#============ Read Config File ============#
num = '2'
alpha = int(sys.argv[1])/10
#alpha = 0.5
#pool_size_upper = int(sys.argv[1])
additional_instance_per_iter = int(sys.argv[2])
pool_size_upper = additional_instance_per_iter*3

#PCA / logits
method_low_dim = 'PCA'

infile = cp.ConfigParser()
infile.read("config/cfg"+num+"gmm.ini")
modelpath = "models/iccad"+num+"gmm/"
datapath  = "benchmarks/fts/data"+num
filename  = "H_set"+num+".data"
fealen=32

maxitr =int(infile.get('al','maxitr'))
phs_co = float(infile.get('al','phs_co'))
additional_instance_num = int(infile.get('al','additional_instance_num'))
#pool_size_upper = int(infile.get('al','pool_size_upper'))
#additional_instance_per_iter = int(infile.get('al','additional_instance_per_iter'))

start_to_active = int(infile.get('al','start_to_active'))
delta1=float(infile.get('al','delta1'))
max_delta1=float(infile.get('al','max_delta1'))
delta2=float(infile.get('al','delta2'))
#ckpt=int(infile.get('al','ckpt'))
#delta1=0
div_channel=int(infile.get('al','div_channel'))
init_idx=datapath+'gmm'+'.csv'  #初始训练数据

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
lr_holder=tf.placeholder(tf.float32, shape=[])                                  #feed in learning rate
#y_gt_c = tf.placeholder(tf.float32, shape=[None, 2])                           
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                   #reshap to NHWC

predict, fea= forward(x, is_training=False)                                     #do forward
loss   = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict)
loss   = tf.reduce_mean(loss)

y      = tf.cast(tf.argmax(predict, 1), tf.int32)                               
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                                                    #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
gs     = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)          #define global step
#lr     =  tf.train.exponential_decay(0.001, gs, decay_steps=20000, decay_rate = 0.65, staircase = True) #initial learning rate and lr decay
lr_base     = 0.0001
lr=lr_base
dr     = 1.0#learning rate decay rate
opt    = tf.train.AdamOptimizer(lr_holder, beta1=0.9)
opt    = opt.minimize(loss, global_step=gs)             #gs用于记录全局训练步骤

bs     = 32                     #training batch size
l_step = 200                    #display step
d_step = 1000                   #lr decay step
test   =1                       #5 for iccad 12  1 for iccad16
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

def ts_plot(trust_score, test_tmp, test_label, times):
    num_per_bin = len(test_label) // pool_size_upper
    data_num = num_per_bin * pool_size_upper
    ts = trust_score[0:data_num]
    acc_ts = np.zeros(pool_size_upper)

    for i in range(pool_size_upper):
        idx = ts.argsort()[num_per_bin*i:num_per_bin*(i+1)]
        acc_ts[i] = sess.run(accu, feed_dict={x_data: test_tmp[idx][:,:,:fealen], y_gt: processlabel(test_label[idx])})
        
    acc = sess.run(accu, feed_dict={x_data: test_tmp[:,:,:fealen], y_gt: processlabel(test_label)})

    x = np.linspace(0,100,pool_size_upper)
    plt.figure()
    plt.plot(x, acc_ts, 'b-', label='Trust Score')
    plt.axvline(x=100*(1-acc), linestyle="dotted", color="black")
    plt.legend(loc='lower right', prop={'size': 11})
    plt.xlabel('Percentile Level')
    plt.ylabel('Precision')
    plt.title('Detect trustworthy'+str(times))
    plt.show()

sess = tf.Session()
#sess.run(tf.global_variables_initializer())
saver    = tf.train.Saver(max_to_keep=400)  #
ckpt = tf.train.get_checkpoint_state(modelpath)

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
'''
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
    #if step == maxitr-1:
    #    path = modelpath
    #    saver.save(sess, path)
    if step % d_step == 0 and step >0:
        lr = lr * dr
lr=lr_base
'''
print("Loading Initial Model...")
saver.restore(sess, "models/initial_model/iccad"+num+"gmm/")

litho_reg  = []
acc_reg = []

for siter in range(sample_iters):
    pool_size = test_data.maxlen
    training_size = train_data.maxlen
    pool_size_per_iter = min(pool_size_upper, pool_size)
    
    test_tmp = test_data.ft_buffer
    train_tmp = train_data.ft_buffer
    test_feature = test_tmp.reshape(test_tmp.shape[0], test_tmp.shape[1]*test_tmp.shape[2])
    test_label = test_data.label_buffer.astype(int)
    train_feature = train_tmp.reshape(train_tmp.shape[0], train_tmp.shape[1]*train_tmp.shape[2])
    train_label = train_data.label_buffer.astype(int)

    print(("STEP[%g/%g] %s: STAT | Principle Component Analysis")%(siter, sample_iters,datetime.now()))
    #nc = min(190, len(test_feature))
    
    nc = pool_size_upper - 5
    pca1 = PCA(n_components=nc)
    pca2 = PCA(n_components=nc)
    pca1.fit(test_feature)
    pca2.fit(train_feature)
    test_feature = pca1.transform(test_feature)
    train_feature = pca2.transform(train_feature)
    
    print(("STEP[%g/%g] %s: STAT | Computing Trust Score")%(siter, sample_iters,datetime.now()))
    if method_low_dim == 'PCA':
        pdt = sess.run(y, feed_dict={x_data: test_tmp[:,:,:fealen]})
        trust_model = TrustScore(k=10, alpha=alpha, filtering="density")
        trust_model.fit(train_feature, train_label)
        trust_score = trust_model.get_score(test_feature, pdt)

    elif method_low_dim == 'logits':
        test_logits = sess.run(predict, feed_dict={x_data: test_tmp[:,:,:fealen]})
        train_logits = sess.run(predict, feed_dict={x_data: train_tmp[:,:,:fealen]})
        pdt = sess.run(y, feed_dict={x_data: test_tmp[:,:,:fealen]})
        trust_model = TrustScore(k=10, alpha=alpha, filtering="density")
        trust_model.fit(train_logits, train_label)
        trust_score = trust_model.get_score(test_logits, pdt)

    indices = trust_score.argsort()[0:additional_instance_per_iter]
    indices_to_drop=trust_score.argsort()[-(pool_size_per_iter-additional_instance_per_iter):][::-1]
    index1 = np.concatenate((indices, indices_to_drop), axis=0)
    #idx_min_ts = np.where(ts < ts_th)

    print(("STEP[%g/%g] %s: STAT | Update Training Set")%(siter, sample_iters,datetime.now()))

    train_data.ft_buffer=np.concatenate((train_data.ft_buffer, test_data.ft_buffer[indices]), axis=0)
    train_data.label_buffer=np.concatenate((train_data.label_buffer, test_data.label_buffer[indices]), axis=0)

    if siter==0:
        drop_data.ft_buffer=test_data.ft_buffer[indices_to_drop]
        drop_data.label_buffer=test_data.label_buffer[indices_to_drop]
    else:
        drop_data.ft_buffer = np.concatenate((drop_data.ft_buffer, test_data.ft_buffer[indices_to_drop]), axis=0)
        drop_data.label_buffer = np.concatenate((drop_data.label_buffer, test_data.label_buffer[indices_to_drop]), axis=0)

    mask = np.ones(len(test_data.label_buffer), dtype=bool)
    mask[index1]=False
    test_data.ft_buffer=test_data.ft_buffer[mask]
    test_data.label_buffer=test_data.label_buffer[mask]
    test_data_for_test.ft_buffer=np.concatenate((drop_data.ft_buffer, test_data.ft_buffer), axis=0)
    test_data_for_test.label_buffer=np.concatenate((drop_data.label_buffer, test_data.label_buffer), axis=0)        

    train_data.reset()
    test_data.reset()
    drop_data.reset()
    test_data_for_test.reset()
    print (train_data.stat())
    print (test_data.stat())
    print (drop_data.stat())
    print (test_data_for_test.stat())

    delta1 = max_delta1*1.0*siter/(sample_iters-1)
    print(("STEP[%g/%g] %s: STAT | Fine Tuning Neural Networks with Delta %f")%(siter, sample_iters,datetime.now(), delta1))
    #if siter==sample_iters-1:
    #    maxitr=10000

    for step in range(maxitr):
        batch = train_data.nextbatch_beta(bs, fealen)
        batch_data = batch[0]
        batch_label= batch[1]

        #batch_label_all_without_bias = processlabel(batch_label, delta1=min(max_delta1, delta1*(1.0*siter/sample_iters)), delta2=min(0.5,delta2*(1.0*siter/sample_iters)))
        batch_label_all_without_bias = processlabel(batch_label, delta1=delta1, delta2=min(0.5,delta2*(1.0*siter/sample_iters)))

        training_loss, training_acc = \
            sess.run(loss, feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias}), sess.run(accu, feed_dict={x_data:batch_data, y_gt:batch_label_all_without_bias})
        opt.run(session=sess, feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias, lr_holder:lr})
        #if step % l_step == 0:
            #format_str = ('         %s: step %d, loss = %.2f, learning_rate = %f, training_accu = %f')
            #print (format_str % (datetime.now(), step, training_loss, learning_rate, training_acc))
        if step == maxitr-1:
            print("                 Saving Checkpoints...")
            path = os.path.join(modelpath, 'al-step-'+str(siter))
            saver.save(sess, path)
        if step % d_step == 0 and step >0:
            lr = lr * dr

    if siter%test==0:
        print(("STEP[%g/%g] STAT | Calculating Missing Hotspots and False Alarms")%(siter, sample_iters))
        chs = 0   #correctly predicted hs
        cnhs= 0   #correctly predicted nhs
        ahs = 0   #actual hs
        anhs= 0   #actual hs
        for titr in range(0, test_data_for_test.maxlen//1000+1):
            if not titr == test_data_for_test.maxlen//1000:
                tbatch = test_data_for_test.nextbatch_without_balance_alpha(1000, fealen)
            else:
                tbatch = test_data_for_test.nextbatch_without_balance_alpha(test_data_for_test.maxlen-titr*1000, fealen)
            tdata = tbatch[0]
            tlabel= tbatch[1].astype(int)
            tmp_y    = sess.run(y, feed_dict={x_data: tdata})

            tmp      = tlabel+tmp_y
            chs += sum(tmp==2)
            cnhs+= sum(tmp==0)
            ahs += sum(tlabel)
            anhs+= sum(tlabel==0)
        if not ahs ==0:
            hs_accu = 1.0*chs/ahs
        else:
            hs_accu = 0
        fs = anhs - cnhs
        acc= (chs+train_data.stat()[0])/(ahs+train_data.stat()[0])
        print('        Total HS count is %g, NHS count is %g'%(train_data.stat()[0]+test_data_for_test.stat()[0], train_data.stat()[1]+test_data_for_test.stat()[1]))
        print('        Total Test HS count is %g, detected count is %g'%(test_data_for_test.stat()[0], chs))
        print('        Current Training Set Count is %g with %g hotspot instances'%(train_data.maxlen,train_data.stat()[0]))
        print('        Missing Hospot is %g'%(ahs-chs))
        print('        Hotspot Accuracy is %f'%acc)
        print('        False Alarm is %f'%fs)
        litho_reg.append(fs+len(train_data.ft_buffer))
        acc_reg.append(acc)

    print(("STEP[%g/%g] STAT | DONE")%(siter, sample_iters))

sess.close()

fn = open("ts5.txt",'a')
#fn.write("ts_th:\t"+str(ts_th)+"\n")
fn.write("select:\t"+str(additional_instance_per_iter)+"\tdrop:\t"+str(additional_instance_per_iter*2)+"\n")
fn.write("alpha:\t"+str(alpha)+"\n")
fn.write("select:\t"+str(pool_size_per_iter)+"\t"+str(additional_instance_per_iter)+"\n")
fn.write("acc:\t"+str(acc_reg)+"\n")
fn.write("litho:\t"+str(litho_reg)+"\n")
fn.write("\n")
fn.close()







