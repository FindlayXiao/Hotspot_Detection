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

num = "3"

infile = cp.ConfigParser()
infile.read("config/cfg"+num+"gmm.ini")
modelpath = "models/iccad"+num+"gmm/"
datapath  = "benchmarks/fts/data"+num
filename  = "H_set"+num+".data"
fealen=32

#fealen=32 #iccad12 comment this line for iccad16
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
init_idx=datapath+'gmm'+'.csv'
phs_co_true=phs_co

data=data(datapath, os.path.join(datapath,'label.csv'), preload=True)

blockdim=np.sqrt(data.ft_buffer.shape[1]).astype(int)
train_data=copy.copy(data)

test_data=copy.copy(data)
test_data_for_test=copy.copy(data)
drop_data=copy.copy(data)
additional_instance_num = min(additional_instance_num, test_data.maxlen)
x_data = tf.placeholder(tf.float32, shape=[None, blockdim*blockdim, fealen])              #input FT
y_gt   = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label
lr_holder=tf.placeholder(tf.float32, shape=[])
#y_gt_c = tf.placeholder(tf.float32, shape=[None, 2])                                      #ground truth label without bias
x      = tf.reshape(x_data, [-1, blockdim, blockdim, fealen])                             #reshap to NHWC

predict, fea= forward(x, is_training=False)                                                            #do forward
loss   = tf.nn.softmax_cross_entropy_with_logits(labels=y_gt, logits=predict)
loss   = tf.reduce_mean(loss)

y      = tf.cast(tf.argmax(predict, 1), tf.int32)
accu   = tf.equal(y, tf.cast(tf.argmax(y_gt, 1), tf.int32))                                                    #calc batch accu
accu   = tf.reduce_mean(tf.cast(accu, tf.float32))
gs     = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)       #define global step
#lr     =  tf.train.exponential_decay(0.001, gs, decay_steps=20000, decay_rate = 0.65, staircase = True) #initial learning rate and lr decay
lr_base     = 0.0001
lr=lr_base
dr     = 1.0#learning rate decay rate
opt    = tf.train.AdamOptimizer(lr_holder, beta1=0.9)
opt    = opt.minimize(loss, gs)

bs     = 32   #training batch size
l_step = 200   #display step
d_step = 1000 #lr decay step
test =1 # 5 for iccad 12  1 for iccad16
stop=0

pool_size_upper = 90
additional_instance_per_iter = 30

additional_instance_num = min(test_data.maxlen, additional_instance_num)
sample_iters = (additional_instance_num-start_to_active)//pool_size_upper
initer= start_to_active/additional_instance_per_iter
phs_co_decay_step=int(sample_iters/1.5)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.45
DEBUG=True
#DEBUG=False

def dis(x, data):
    D = []
    for i in range(0, len(data)):
        D.append(np.linalg.norm(x - data[i]))
    result = sum(D) / len(data)
    return result

def idx_a(H_set, alpha):
    rk = np.zeros(len(H_set))
    for i in range(0, len(H_set)):
        if len(H_set) < 10:
            idx = H_set[i].argsort()[-1]
        else:
            idx = H_set[i].argsort()[9]
        rk[i] = H_set[i][idx]
    idx_h = rk.argsort()[:round(alpha*len(H_set))]
    return np.array(idx_h)

# Read H_set from file
f = open(filename, "rb")
H_set = pickle.load(f)
f.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=400)
ckpt = tf.train.get_checkpoint_state(modelpath)

index_test = np.genfromtxt(init_idx, delimiter=',').astype(int)
mask = np.ones(len(test_data.label_buffer), dtype=bool)
mask[index_test]=False
test_data.ft_buffer=test_data.ft_buffer[index_test]
test_data.label_buffer=test_data.label_buffer[index_test]
H_set = H_set[np.ix_(index_test, index_test)]
test_data_for_test.ft_buffer=test_data_for_test.ft_buffer[index_test]
test_data_for_test.label_buffer=test_data_for_test.label_buffer[index_test]

train_data.ft_buffer=train_data.ft_buffer[mask]
train_data.label_buffer=train_data.label_buffer[mask]

train_data.reset()
test_data.reset()  
test_data_for_test.reset()     
print (train_data.stat())
print (test_data.stat())

print("Training Initialize Model......")
for step in range(maxitr*4):
    batch = train_data.nextbatch_beta(bs, fealen)
    batch_data = batch[0]
    batch_label= batch[1]

    batch_label_all_without_bias = processlabel(batch_label, delta1=0)

    training_loss, training_acc = \
        sess.run(loss, feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias}), sess.run(accu, feed_dict={x_data:batch_data, y_gt:batch_label_all_without_bias})
    opt.run(session=sess, feed_dict={x_data: batch_data, y_gt: batch_label_all_without_bias, lr_holder: lr})
    if step % l_step == 0:
        format_str = ('%s: step %d, loss = %.2f, training_accu = %f')
        print (format_str % (datetime.now(), step, training_loss, training_acc))
    #if step == maxitr-1:
    #    path = modelpath
    #    saver.save(sess, path)
    if step % d_step == 0 and step >0:
        lr = lr * dr
lr=lr_base

for siter in range(sample_iters):
    pool_size = test_data.maxlen
    training_size = train_data.maxlen

    pool_size_per_iter = min(pool_size_upper, pool_size)
    #pool_idxs_per_iter = np.array(random.sample(range(0, pool_size), pool_size_per_iter))
    test_feature=test_data.ft_buffer
    test_label=test_data.label_buffer
    H_loop = H_set

    print(("STEP[%g/%g] %s: STAT | Querying Feature Vectors and Uncertainty Posterior")%(siter, sample_iters,datetime.now()))
    #uncertainty=predict.eval(feed_dict={x_data: test_feature[pool_idxs_per_iter][:,:,:fealen]})
    #feature=fea.eval(feed_dict={x_data: test_feature[pool_idxs_per_iter][:,:,:fealen]})

    prob = sess.run(predict, feed_dict={x_data: test_feature[:,:,:fealen]})
    hs_score=prob[:,1]
    index1=hs_score.argsort()[-pool_size_per_iter:][::-1]
    #index1 = np.arange(0,pool_size_per_iter)
    # use trust_score to select samples
    print(("STEP[%g/%g] %s: STAT | Computing Trust Score")%(siter, sample_iters,datetime.now()))
    pdt = sess.run(y, feed_dict={x_data: test_feature[index1][:,:,:fealen]})
    H_ts = H_loop[np.ix_(index1, index1)]

    alpha = 0.3
    k = 10
    idx_p = np.nonzero(pdt != 0)[0]
    idx_n = np.nonzero(pdt == 0)[0]
    idx_p2= np.ix_(idx_p, idx_p)
    idx_n2= np.ix_(idx_n, idx_n)

    idx_a_n = idx_a(H_ts[idx_n2], alpha)
    idx_a_p = idx_a(H_ts[idx_p2], alpha)

    H1 = H_ts[idx_n[idx_a_n]]
    H2 = H_ts[idx_p[idx_a_p]]

    ts = np.zeros(len(index1))
    for i in range(0, len(index1)):
        if pdt[i] == 0:
            H_x   = H1
            H_x_n = H2
        else: 
            H_x   = H2
            H_x_n = H1
        if len(H_x_n)==0 or len(H_x)==0:
            ts[i] = 0
        else:
            ts[i] = H_x_n[:,i].min() / H_x[:,i].min()

    idx_min_ts = ts.argsort()[0:additional_instance_per_iter]
    #idx_min_ts = np.arange(0,additional_instance_per_iter)

    mask0=np.ones(len(index1), dtype=bool)
    mask0[idx_min_ts]=False
    indices=index1[idx_min_ts]
    indices_to_drop=index1[mask0]                   # 丢弃未被选择的部分
    if np.sum(test_data.label_buffer[indices])==0:
        phs_co_true=0
    else:
        phs_co_true=phs_co

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
    H_set = H_set[np.ix_(mask, mask)]
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
        print('        Total HS count is %g, NHS count is %g'%(train_data.stat()[0]+test_data_for_test.stat()[0], train_data.stat()[1]+test_data_for_test.stat()[1]))
        print('        Total Test HS count is %g, detected count is %g'%(test_data_for_test.stat()[0], chs))
        print('        Current Training Set Count is %g with %g hotspot instances'%(train_data.maxlen,train_data.stat()[0]))
        print('        Missing Hospot is %g'%(ahs-chs))
        print('        Hotspot Accuracy is %f'%((chs+train_data.stat()[0])/(ahs+train_data.stat()[0])))
        print('        False Alarm is %f'%fs)


    print(("STEP[%g/%g] STAT | DONE")%(siter, sample_iters))

sess.close()