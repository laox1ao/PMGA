#!/usr/bin/env python
#-*-coding:utf-8-*-

from __future__ import print_function
import numpy as np
import os
import sys
import argparse
import random
from model.base_line import Base_Line
from tools.dataprocess import DataGenerator as DG
from tools.dataprocess import evaluate_map_mrr, evaluate_score
from tools import data_process as d_p
import tensorflow as tf
#from model.myModel import _Model

random.seed(1337)

parser = argparse.ArgumentParser(description="QA2018")
parser.add_argument('-m','--model',type=str,default='base_line',help="base line model")
parser.add_argument('-b','--batch_size',type=int,default=1,help="batch size")
parser.add_argument('-ls','--list_size',type=int,default=15,help="list-wise size")
parser.add_argument('-e','--epochs',type=int,default=8,help="train epochs")
parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help="learning rate")
parser.add_argument('-hd','--hidden_dim',type=int,default=300,help="hidden dim")
parser.add_argument('-gid','--gpu_id',type=int,default=1,help="gpu id")
parser.add_argument('-ql','--ques_len',type=int,default=25,help="question length")
parser.add_argument('-al','--ans_len',type=int,default=90,help="question length")
parser.add_argument('-cv','--cv',type=int,default=1,help="cv num")

current_path = os.path.abspath(".")
train_file = current_path+"/data/wikiqa/wiki_train.pkl"
dev_file = current_path+"/data/wikiqa/wiki_dev.pkl"
test_file = current_path+"/data/wikiqa/wiki_test.pkl"
answer_file = current_path+"/data/wikiqa/wiki_answer_train.pkl"
embedding_file = current_path+"/data/wikiqa/wikiqa_glovec.txt"

def main(args):
    ######model_params
    model_params = Model_Param(args)
    model_params.print_param()

    dg = DG(model_params)
    dg_ = d_p.DataGenerator(1,model_params,'./data/wikiqa/wiki_answer_train.pkl')

    #####data
    print("train_data:")
    #train_data = dg.data_listwise_clean(train_file,answer_file)
    train_data = dg_.wikiQaGenerate(train_file)
    print("dev_data:")
    dev_data = dg.test_listwise_clean(dev_file)
    print("test_data:")
    test_data = dg.test_listwise_clean(test_file)

    train_steps = len(train_data)/model_params.batch_size+1

    #####model
    model = Base_Line(model_params)
    model._build_base_line_listwise()
    #score_op = model.score
    #loss_op = model.loss_listwise

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    best_dev_map, best_test_map = 0.0, 0.0
    best_dev_epoch, best_test_epoch = 0, 0

    loss_op = model.loss_listwise
    optimizer = tf.train.AdamOptimizer(learning_rate=model_params.learning_rate)
    train_op = optimizer.minimize(loss_op)
    sess.run(tf.global_variables_initializer())
    for i in range(1,model_params.epochs+1):
        print("===================================Epoch %s================================" % i)
        train_data_ = zip(*train_data)
        np.random.shuffle(train_data_)
        train_data = map(np.array,zip(*train_data_))
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
                                                              model.l_label:batch_l_l
                                                              })
            loss_e += loss_b
        loss_e /= train_steps

        dev_label = dev_data[4]
        test_label = test_data[4]

        score_list = evaluate_score(sess,model,dev_data)
        dev_mAp, dev_mRr = evaluate_map_mrr(score_list,dev_label)

        score_list = evaluate_score(sess,model,test_data)
        test_mAp, test_mRr = evaluate_map_mrr(score_list,test_label)

        if(dev_mAp>best_dev_map):
            best_dev_map = dev_mAp
            best_dev_mrr = dev_mRr
            best_dev_epoch = i
        if(test_mAp>best_test_map):
            best_test_map = test_mAp
            best_test_mrr = test_mRr
            best_test_epoch = i

        print("Loss at epoch %s: %.4f\n" % (i,loss_e))
        print("MAP on dev_data: %.4f,\t\t" %  dev_mAp,"MRR on dev_data: %.4f\n" % dev_mRr)
        print("MAP on test_data: %.4f,\t\t" % test_mAp,"MRR on test_data: %.4f\n" % test_mRr)

    print("===============================\non dev Best MAP: %.4f\t\tBest MRR: %.4f\tat epoch %s" % (best_dev_map,best_dev_mrr,best_dev_epoch),
          "\non test Best MAP: %.4f\tBest MRR: %.4f\tat epoch %s\n===============================" % (best_test_map,best_test_mrr,best_test_epoch))
    return best_test_map, best_test_mrr

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
    def print_param(self):
        print("model: ",self.model,
              "\nbatch_size: ",self.batch_size,
              "\nlist_size: ",self.list_size,
              "\nepochs: ",self.epochs,
              "\nlearning_rate: ",self.learning_rate,
              "\nhidden_dim: ",self.hidden_dim,
              "\nques_len: ",self.ques_len,
              "\nans_len: ",self.ans_len)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    ave_map, ave_mrr = 0.0, 0.0
    for i in range(args.cv):
        mAp, mRr = main(args)
        ave_map += mAp
        ave_mrr += mRr
    ave_map /= args.cv
    ave_mrr /= args.cv
