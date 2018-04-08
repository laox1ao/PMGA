#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf
import numpy as np
from collections import namedtuple
import pickle
import random

class DataGenerator(object):
    def __init__(self,params):
        self.params = params

    def padseq(self,seq_to_pad,pad_to_len):
        if(len(seq_to_pad)>=pad_to_len):
            return seq_to_pad[:pad_to_len]
        else:
            seq_to_pad.extend([0 for i in range(pad_to_len-len(seq_to_pad))])
            return seq_to_pad

    def data_pointwise(self,train_file):
        train_f = pickle.load(open(train_file,'r'))
        train_size = len(train_f)
        question, answer, label = zip(*train_f)
        question_len= map(lambda x: len(x),question)
        answer_len = map(lambda x: len(x),answer)
        question = map(lambda x: self.padseq(x,self.params.ques_len),question)
        answer = map(lambda x: self.padseq(x,self.params.ques_len),answer)

        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_len = np.array(question_len)
        answer_len = np.array(answer_len)
        print question.shape
        print question_len.shape
        print answer.shape
        print answer_len.shape
        print label.shape

        return question, question_len, answer, answer_len, label

    def data_listwise(self,train_file,answer_file,list_size=15):
        train_f = pickle.load(open(train_file,'r'))
        answer_f = pickle.load(open(answer_file,'r'))
        train_size = len(train_f)
        question,answer,label = zip(*train_f)
        assert len(question)==len(answer)==len(label), "Invalid train data with vary size among question,answer,label!"
        question_len = map(lambda x: [1 for _ in range(len(x))], question)
        answer_len = map(lambda x: [1 for _ in range(len(x))], answer)
        train_dic = dict()
        for i,ques in enumerate(question):
            train_dic.setdefault(str(ques),[])
#            if(label[i]==1):
#                train_dic[str(ques)][0] += 1
            train_dic[str(ques)].append([ques,answer[i],question_len[i],answer_len[i],label[i]])
        print "size of train_dic:",len(train_dic)

        questions = []
        answers = []
        questions_len = []
        answers_len = []
        labels = []
        for k,v in train_dic.items():
            if(len(v)>=list_size):
                false_sample = [i for i in range(len(v)) if v[i][-1]==0]
                filtered_false_sample = set(random.sample(false_sample,len(v)-list_size))
                train_dic[k] = [v[i] for i in range(len(v)) if i not in filtered_false_sample]
            else:
                pad_size = list_size - len(v)
                pad_answer = random.sample(answer_f.values(),pad_size)
                pad_answer_len = [[1 for _ in range(len(x))] for x in pad_answer]
                pad_sample = [[v[0][0],pad_answer[i],v[0][2],pad_answer_len[i],0] for i in range(len(pad_answer))]
                train_dic[k].extend(pad_sample)
            ques, ans, que_len, ans_len, label = zip(*train_dic[k])
            ques = map(lambda x: self.padseq(x,self.params.ques_len),ques)
            ans = map(lambda x: self.padseq(x,self.params.ans_len),ans)
            ques_len = map(lambda x: self.padseq(x,self.params.ques_len),ques)
            ans_len = map(lambda x: self.padseq(x,self.params.ans_len),ans)
            questions.append(ques)
            answers.append(ans)
            questions_len.append(ques_len)
            answers_len.append(ans_len)
            labels.append(label)
        questions = np.array(questions)
        answers = np.array(answers)
        questions_len = np.array(questions_len)
        answers_len = np.array(answers_len)
        labels = np.array(labels)

        print questions.shape
        print answers.shape
        print questions_len.shape
        print answers_len.shape
        print labels.shape

        return questions, answers, questions_len,answers_len,labels
#            train_dic[k] = zip(*[ques,ans,ques_len,ans_len,label])




    def dev_pointwise(self,dev_file):
        dev_f = pickle.load(open(dev_file,'r'))
        dev_size = len(dev_f)
        question, answer, label = zip(*dev_f)
        question_len= map(lambda x: len(x),question)
        answer_len = map(lambda x: len(x),answer)
        self.padseq(question,self.params.ques_len)
        self.padseq(answer,self.params.ans_len)

        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_len = np.array(question_len)
        answer_len = np.array(answer_len)
        print question.shape
        print question_len.shape
        print answer.shape
        print answer_len.shape
        print label.shape

        return question, question_len, answer, answer_len, label

    def test_pointwise(self,test_file):
        test_f = pickle.load(open(test_file,'r'))
        test_size = len(test_f)
        question, answer, label = zip(*dev_f)
        question_len= map(lambda x: len(x),question)
        answer_len = map(lambda x: len(x),answer)
        self.padseq(question,self.params.ques_len)
        self.padseq(answer,self.params.ans_len)

        question = np.array(question)
        answer = np.array(answer)
        label = np.array(label)
        question_len = np.array(question_len)
        answer_len = np.array(answer_len)
        print question.shape
        print question_len.shape
        print answer.shape
        print answer_len.shape
        print label.shape

        return question, question_len, answer, answer_len, label

if __name__ == '__main__':
    class Param():
        def __init__(self):
            self.ques_len = 10
            self.ans_len = 20
    param = Param()
    datag = DataGenerator(param)
#    datag.train_pointwise('../data/wikiqa/wiki_train.pkl')
    datag.data_listwise('../data/wikiqa/wiki_dev.pkl','../data/wikiqa/wiki_answer_train.pkl')
    datag.data_listwise('../data/wikiqa/wiki_test.pkl','../data/wikiqa/wiki_answer_train.pkl')

