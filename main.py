#!/usr/bin/env python
#-*-coding:utf-8-*-

from __future__ import print_function
import numpy as np
import os
import sys
import argparse
import time
import random
from model.base_line import Base_Line
from model.my_model import My_Model
from tools.dataprocess import DataGenerator as DG
from tools.dataprocess import evaluate_map_mrr, evaluate_score
from tools import data_process as d_p
import tensorflow as tf
#from model.myModel import _Model

#random.seed(1337)
#np.random.seed(1337)
#tf.set_random_seed(1337)

parser = argparse.ArgumentParser(description="QA2018")
parser.add_argument('-m','--model',type=str,default='b',help="base line model")
parser.add_argument('-b','--batch_size',type=int,default=3,help="batch size")
parser.add_argument('-ls','--list_size',type=int,default=15,help="list-wise size")
parser.add_argument('-e','--epochs',type=int,default=200,help="train epochs")
parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help="learning rate")
parser.add_argument('-hd','--hidden_dim',type=int,default=300,help="hidden dim")
parser.add_argument('-gid','--gpu_id',type=str,default='0,1',help="gpu id")
parser.add_argument('-ql','--ques_len',type=int,default=25,help="question length")
parser.add_argument('-al','--ans_len',type=int,default=90,help="question length")
parser.add_argument('-cv','--cv',type=int,default=1,help="cv num")
parser.add_argument('-kp','--keep_prob',type=float,default=0.0,help="keep prob")
parser.add_argument('-lrdc','--lr_decay',type=float,default=0.99,help="learning rate decay")
parser.add_argument('-lrsr','--lr_shrink',type=float,default=0.5,help="learning rate shrink")
parser.add_argument('-opt','--optimizer',type=str,default='adam',help="optimizer")

current_path = os.path.abspath(".")
train_file = current_path+"/data/wikiqa/wiki_train.pkl"
dev_file = current_path+"/data/wikiqa/wiki_dev.pkl"
test_file = current_path+"/data/wikiqa/wiki_test.pkl"
answer_file = current_path+"/data/wikiqa/wiki_answer_train.pkl"
embedding_file = current_path+"/data/wikiqa/wikiqa_glovec.txt"

def get_learning_rate(lr,model_params,dev_map,last_dev_map,epoch):
    if epoch<25:
        return lr
    if dev_map>=last_dev_map:
        return lr*model_params.lr_decay
    else:
        return lr*model_params.lr_shrink


def main(args):
    ave_dev_map, ave_dev_mrr = 0.0, 0.0
    ave_dev_test_map, ave_dev_test_mrr = 0.0, 0.0
    ave_test_map, ave_test_mrr = 0.0, 0.0
    with tf.variable_scope('cv',reuse=tf.AUTO_REUSE):
        ######model_params
        model_params = Model_Param(args)
        model_params.print_param()

        #####model
        if model_params.model == 'b':
            model_type = 'baseline'
            model = Base_Line(model_params)
            model._build_base_line_listwise()
        elif model_params.model == 'm':
            model_type = 'syn_ext'
            model = My_Model(model_params)
            model._build_syn_ext_listwise()

        dg = DG(model_params)
        dg_ = d_p.DataGenerator(1,model_params,'./data/wikiqa/wiki_answer_train.pkl')
        for c in range(args.cv):
            #####data
            print("dev_data:")
            dev_data = dg.test_listwise_clean(dev_file)
            print("test_data:")
            test_data = dg.test_listwise_clean(test_file)

            #score_op = model.score
            #loss_op = model.loss_listwise

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            best_dev_map, best_test_map = 0.0, 0.0
            dev_test_map, dev_test_mrr = 0.0, 0.0
            best_dev_epoch, best_test_epoch = 0, 0

            global_step = tf.Variable(0,name='global_step',trainable=False)
            #learning_rate = tf.train.exponential_decay(0.0005,global_step,873*3,0.9,staircase=True)
            learning_rate_op = tf.placeholder(tf.float32)
            learning_rate = model_params.learning_rate
            loss_op = model.loss_listwise
            if(model_params.optimizer=='adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate_op)
            elif(model_params.optimizer=='adadelta'):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate_op)
            elif(model_params.optimizer=='sgd'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate_op)
            train_op = optimizer.minimize(loss_op,global_step=global_step)
            last_dev_mAp = 0.0
            sess.run(tf.global_variables_initializer())
            for i in range(1,model_params.epochs+1):
                print("===================================CV %s Epoch %s================================" % (c,i))
                print("train_data:")
                #train_data = dg.data_listwise_clean(train_file,answer_file)
                #train_data = dg.data_listwise_clean_internal_sample(train_file,answer_file)
                train_data = dg_.wikiQaGenerate(train_file)
                print("learnig_rate---> %s" % learning_rate)
                train_data_ = zip(*train_data)
                np.random.shuffle(train_data_)
                train_data = map(np.array,zip(*train_data_))
                train_steps = len(train_data)/model_params.batch_size+1
                loss_e = 0.0
                for j in range(train_steps):
                    batch_ques = train_data[0][j*model_params.batch_size:(j+1)*model_params.batch_size,:,:]
                    batch_ans = train_data[1][j*model_params.batch_size:(j+1)*model_params.batch_size,:,:]
                    batch_ques_l = train_data[2][j*model_params.batch_size:(j+1)*model_params.batch_size,:,:]
                    batch_ans_l = train_data[3][j*model_params.batch_size:(j+1)*model_params.batch_size,:,:]
                    #batch_p_l = train_data[4][j*model_params.batch_size:(j+1)*model_params.batch_size,:,:]
                    batch_l_l = train_data[4][j*model_params.batch_size:(j+1)*model_params.batch_size,:]

                    loss_b, _ = sess.run([loss_op,train_op], feed_dict={model._ques:batch_ques,
                                                                      model._ans:batch_ans,
                                                                      model._ques_len:batch_ques_l,
                                                                      model._ans_len:batch_ans_l,
                                                                      model.l_label:batch_l_l,
                                                                        model.is_train:True,
                                                                        learning_rate_op:learning_rate
                                                                      })
                    current_sam = j*model_params.batch_size+len(batch_ans)
                    total_sam = len(train_data[1])
                    #c_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
                    #print(c_time+'['+'='*20+'] '+' loss: %.8f' % loss_b)
                    loss_e += loss_b
                loss_e /= train_steps

                dev_label = dev_data[4]
                test_label = test_data[4]

                score_list = evaluate_score(sess,model,dev_data)
                dev_mAp, dev_mRr = evaluate_map_mrr(score_list,dev_label)
                #learning_rate = get_learning_rate(learning_rate,model_params,dev_mAp,last_dev_mAp,i)
                last_dev_mAp = dev_mAp

                score_list = evaluate_score(sess,model,test_data)
                test_mAp, test_mRr = evaluate_map_mrr(score_list,test_label)

                if(dev_mAp>best_dev_map):
                    best_dev_map = dev_mAp
                    best_dev_mrr = dev_mRr
                    dev_test_map = test_mAp
                    dev_test_mrr = test_mRr
                    best_dev_epoch = i
                if(test_mAp>best_test_map):
                    best_test_map = test_mAp
                    best_test_mrr = test_mRr
                    best_test_epoch = i

                print("Loss at epoch %s: %.4f\n" % (i,loss_e))
                print("MAP on dev_data: %.4f,\t\t" %  dev_mAp,"MRR on dev_data: %.4f\n" % dev_mRr)
                print("MAP on test_data: %.4f,\t\t" % test_mAp,"MRR on test_data: %.4f\n" % test_mRr)

            print("===============================\non dev Best MAP: %.4f\t\tBest MRR: %.4f\tat epoch %s" % (best_dev_map,best_dev_mrr,best_dev_epoch),
                  "\non dev test MAP: %.4f\t\ttest MRR: %.4f\tat epoch %s\n===============================" % (dev_test_map,dev_test_mrr,best_dev_epoch),
                  "\non test Best MAP: %.4f\tBest MRR: %.4f\tat epoch %s\n===============================" % (best_test_map,best_test_mrr,best_test_epoch))
            if(best_test_map>ave_test_map): ave_test_map = best_test_map;ave_test_mrr = best_test_mrr
            if(best_dev_map>ave_dev_map): ave_dev_map = best_dev_map;ave_dev_mrr = best_dev_mrr
            if(dev_test_map>ave_dev_test_map): ave_dev_test_map = dev_test_map;ave_dev_test_mrr = dev_test_mrr
            #ave_test_map += best_test_map
            #ave_test_mrr += best_test_mrr
            #ave_dev_map += best_dev_map
            #ave_dev_mrr += best_dev_mrr
            #ave_dev_test_map += dev_test_map
            #ave_dev_test_mrr += dev_test_mrr
        #ave_dev_map /= args.cv
        #ave_dev_mrr /= args.cv
        #ave_test_map /= args.cv
        #ave_test_mrr /= args.cv
        #ave_dev_test_map /= args.cv
        #ave_dev_test_mrr /= args.cv
        print("\n[==Model %s Finished After %s CV==]\non dev Average MAP : %.4f\tAverage MRR : %.4f\non dev test Average MAP: %.4f\tAverage MRR : %.4f\non test Average MAP : %.4f\tAverage MRR : %.4f"% (model_type,args.cv,ave_dev_map,ave_dev_mrr,ave_dev_test_map,ave_dev_test_mrr,ave_test_map,ave_test_mrr))
class Model_Param():
    def __init__(self,args):
        self.model = args.model
        self.batch_size = args.batch_size
        self.list_size = args.list_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.hidden_dim = args.hidden_dim
        self.ques_len = args.ques_len
        self.ans_len = args.ans_len
        self.embedding_file = embedding_file
        self.random_size = args.list_size
        self.keep_prob = args.keep_prob
        self.lr_decay = args.lr_decay
        self.lr_shrink = args.lr_shrink
        self.optimizer = args.optimizer
    def print_param(self):
        print("model: ",self.model,
              "\nbatch_size: ",self.batch_size,
              "\nlist_size: ",self.list_size,
              "\nepochs: ",self.epochs,
              "\nlearning_rate: ",self.learning_rate,
              "\nkeep_prob: ",self.keep_prob,
              "\nhidden_dim: ",self.hidden_dim,
              "\nques_len: ",self.ques_len,
              "\nans_len: ",self.ans_len)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)
