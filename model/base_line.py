#!/usr/bin/env python
#-*-coding:utf-8-*-

from __future__ import print_function
import numpy as np
import tensorflow as tf

class Base_Line():
    def __init__(self):
        pass

    def _build_base_line_pointwise(self):
        with tf.variable_scope('inputs') as inputs:
            self._ques = tf.placeholder(tf.float32,[self.batch_size,self.ques_len],name='ques_point')
            self._ques_len = tf.placeholder(tf.float32,[self.batch_size,self.ques_len],name='ques_len_point')
            self._ans = tf.placeholder(tf.float32,[self.batch_size,self.ans_len],name='ans_point')
            self._ans_len = tf.placeholder(tf.float32,[self.batch_size,self.ans_len],name='ans_len_point')

            self._ques_filter_len = tf.tile(tf.reshape(self._ans_len,[-1,1,self.ans_len]),[1,self.ques_len,1])
            self._ans_filter_len = tf.tile(tf.reshape(self._ques_len,[-1,1,self.ques_len]),[1,self.ans_len,1])

            self._ques_align_len = tf.tile(tf.reshape(self._ques_len,[-1,self.ques_len,1]),[1,1,self.hidden_dim])
            self._ans_align_len = tf.tile(tf.reshape(self._ans_len,[-1,self.ans_len,1]),[1,1,self.hidden_dim])
        with tf.name_scope('point_wise'):
            with tf.variable_scope('embedding_layer') as embedding_l:
                weights = np.load(self.embedding_file)
                weights[0] = np.zero((weights.shape[1]))
                embeddings = tf.Variable(weights,dtype=float32)

                ques_emb = tf.nn.embedding_lookup(embeddings,self._ques)
                ans_emb = tf.nn.embedding_lookup(embeddings,self._ans)
            with tf.variable_scope('preprocess_layer') as prep_l:
                sig_den = tf.layers.Dense(self.hidden_dim,activation=tf.sigmoid,name='sigmoid_dense')
                tan_den = tf.layers.Dense(self.hidden_dim,activation=tf.tanh,name='tanh_dense')

                ques_sig = sig_den(ques_emb)
                ques_tan = tan_den(ques_emb)
                ques_h = tf.multiply(ques_sig,ques_tan)

                ans_sig = sig_den(ans_emb)
                ans_tan = tan_den(ans_emb)
                ans_h = tf.multiply(ans_sig,ans_tan)
            with tf.variable_scope('attention_softalign') as att_align_l:


    def _build_base_line_listwise(self):
        with tf.variable_scope('inputs') as inputs:
            self._ques = tf.placeholder(tf.float32,[self._batch_size,self.list_size,self.ques_len],name='ques')
            self._ques_len = tf.placeholder(tf.float32,[self._batch_size,self.list_size,self.ques_len],name='ques_len')
            self._ans = tf.placeholder(tf.float32,[self._batch_size,self.list_size,self.ans_len],name='ans')
            self._ans_len = tf.placeholder(tf.float32,[self._batch_size,self.list_size,self.ans_len],name='ans_len')
        with tf.name_scope('list_wise') as list_wise:
            with tf.variable_scope('_list_wise'):
                for i in range(self.list_size):
                    self._ques_point = self._ques[:,i,:]
                    self._ques_len_point = self._ques_len[:,i,:]
                    self._ans_point = self._ans[:,i,:]
                    self._ans_len_point = self._ans_len[:,i,:]
                    self.score = self._build_base_line_pointwise()

