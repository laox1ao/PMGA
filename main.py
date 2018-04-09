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
from tools.dataprocess import evaluate_map_mrr
import tensorflow as tf
#from model.myModel import _Model

random.seed(1337)

parser = argparse.ArgumentParser(description="QA2018")
parser.add_argument('-m','--model',type=str,default='base_line',help="base line model")
parser.add_argument('-b','--batch_size',type=int,default=10,help="batch size")
parser.add_argument('-ls','--list_size',type=int,default=15,help="list-wise size")
parser.add_argument('-e','--epochs',type=int,default=8,help="train epochs")
parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help="learning rate")
parser.add_argument('-hd','--hidden_dim',type=int,default=300,help="hidden dim")
parser.add_argument('-gid','--gpu_id',type=int,default=0,help="gpu id")
parser.add_argument('-ql','--ques_len',type=int,default=25,help="question length")
parser.add_argument('-al','--ans_len',type=int,default=90,help="question length")

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

    #####data
    print("train_data:")
    train_data = dg.data_listwise_wo0(train_file,answer_file)
    print("dev_data:")
    dev_data = dg.data_listwise_wo0(dev_file,answer_file)
    print("test_data:")
    test_data = dg.test_listwise(test_file)

    train_steps = len(train_data)/model_params.batch_size+1

    #####model
    model = Base_Line(model_params)
    loss_op = model.loss_listwise
    optimizer = tf.train.AdamOptimizer(learning_rate=model_params.learning_rate)
    train_op = optimizer.minimize(loss_op)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    best_map, best_mrr = 0.0, 0.0
    best_epoch = 0

    for i in range(1,model_params.epochs+1):
        print("===================================Epoch %s================================" % i)
        map(np.random.shuffle,train_data)
        for j in range(train_steps):
            batch_ques = train_data[0][j*model_params.batch_size:(j+1)*model_params.batch_size]
            batch_ans = train_data[1][j*model_params.batch_size:(j+1)*model_params.batch_size]
            batch_ques_l = train_data[2][j*model_params.batch_size:(j+1)*model_params.batch_size]
            batch_ans_l = train_data[3][j*model_params.batch_size:(j+1)*model_params.batch_size]
            batch_p_l = train_data[4][j*model_params.batch_size:(j+1)*model_params.batch_size]
            batch_l_l = train_data[5][j*model_params.batch_size:(j+1)*model_params.batch_size]

            _, loss_train = sess.run([train_op,loss_op],feed_dict={model._ques:batch_ques, model._ans:batch_ans,
                                         model._ques_len:batch_ques_l, model._ans_len:batch_ans_l,
                                         model.p_label:batch_p_l, model.l_label: batch_l_l})
        dev_ques = dev_data[0]
        dev_ans = dev_data[1]
        dev_ques_l = dev_data[2]
        dev_ans_l = dev_data[3]
        dev_p_l = dev_data[4]
        dev_l_l = dev_data[5]

        loss, score = sess.run([loss_op,model.score],feed_dict={model._ques:dev_ques, model._ans:dev_ans,
                                                                model._ques_len:dev_ques_l, model._ans_len:dev_ans_l,
                                                                model.p_label:dev_p_l, model.l_label:dev_l_l})
        score = score[:,:,0]
        print(score.shape)
        print(dev_p_l.shape)

        mAp, mRr = evaluate_map_mrr(score,dev_p_l)
        if(mAp>best_map):
            best_map = mAp
            best_mrr = mRr
            best_epoch = i

        print("Loss at epoch %s: %.4f\n" % (i,loss_train))
        print("MAP on dev_data: %.4f\n" % mAp,"MRR on dev_data: %.4f\n" % mRr)

    print("===============================\nBest MAP: %.4f\nBest MRR: %.4f\nat epoch %s" % (best_map,best_mrr,best_epoch))

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
    main(args)
