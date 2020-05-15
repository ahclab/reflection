#!/usr/bin/env python
import sys

import chainer
from chainer import serializers
from chainer import Variable

import pandas as pd
import numpy as np
import argparse
import gensim
import json

import nets
import data_loader as data


def reset_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
        

def get_sentence():
    try:
        inp = input('Input:\t').split(' ')
        return inp
    except ValueError:
        return get_sentence()

    
def get_attr_id():
    try:
        attr_id = int(input('Attribute ID (-2:w/o-net, -1:end, 0:gender): '))
        if attr_id==-1:
            sys.exit()
        return attr_id
    except ValueError:
        return get_attr_id()

    

def show_result(model, dtype):
    xs = model['setting']['accuracy'][dtype]['0']['x']
    ts = model['setting']['accuracy'][dtype]['0']['t']
    for i, x in enumerate(xs):
        acc = model['setting']['accuracy'][dtype]['0']['acc@1'][i]
        y = model['setting']['accuracy'][dtype]['0']['nearest_words'][i][0][0]
        print(acc, x, ts[i], y)
        
        
        
class Analogy():
    def __init__(self, d, operation):
        # e.g., d = 'himself-herself'
        self.m = d.split('-')[0]
        self.w = d.split('-')[1]
        self.operation = operation
        self.net_name = 'analogy'

            
    def test(self, x, word2vec):
        if self.operation=='+':
            return x + (word2vec[self.m] - word2vec[self.w])
        elif self.operation=='-':
            return x - (word2vec[self.m] - word2vec[self.w])
        else:
            assert False
    
    
    
class Demo():
    def __init__(self, word2vec):
        self.word2vec = word2vec 
        
    def get_model(self, path_setting, path_netdir):
        # Load setthings of the model
        #path_setting = '{}/{}/evaluation_results.json'.format(path_models, model_type)
        with open(path_setting, 'r') as f:
            setting = json.load(f)   
            
        model_type = setting['experimental_settings']['model']['name']
        dim_x = setting['experimental_settings']['model']['dim_x']
        n_attribute = setting['experimental_settings']['model']['n_attribute']
        sigma = setting['experimental_settings']['model']['sigma']
        if 'Ref' in model_type:
            dim_h = setting['experimental_settings']['model']['dim_h']
            
        # Load a net
        path_net = '{}/{}'.format(path_netdir, setting['net']['best'].split('/')[-1]) # select best net from result file
        
        if 'Ref+PM' in model_type:
            net = nets.Ref_PM(dim_x, dim_h, n_attribute, sigma)
        elif 'Ref' in model_type:
            net = nets.Ref(dim_x, dim_h, n_attribute, sigma)
        elif 'MLP1' in model_type:
            net = nets.MLP_v1(dim_x, n_attribute, sigma)
        elif 'MLP2' in model_type:
            net = nets.MLP_v2(dim_x, n_attribute, sigma)
        elif 'MLP3' in model_type:
            net = nets.MLP_v3(dim_x, n_attribute, sigma)
        
        serializers.load_npz(path_net, net)
    
        return {'net':net, 'setting':setting, 'model_type':model_type}
    
    
    def get_analogy(self, path_setting):
        #path_setting = '{}/{}/evaluation_results.json'.format(path_models, model_type)
        with open(path_setting, 'r') as f:
            setting = json.load(f)   
            
        if setting['experimental_settings']['model']['name']=='one_diff':
            model_type = 'Diff'
        elif setting['experimental_settings']['model']['name']=='mean_diff':
            model_type = 'MeanDiff'
        if setting['experimental_settings']['model']['plus_minus_or_info']=='plus-all':
            model_type += '+'
        elif setting['experimental_settings']['model']['plus_minus_or_info']=='minus-all':
            model_type += '-'
            
        if 'Diff+' in model_type:
            d = setting['best_result']['pair']
            net = Analogy(d, '+')
        if 'Diff-' in model_type:
            d = setting['best_result']['pair']
            net = Analogy(d, '-')
            
        return {'net':net, 'setting':setting, 'model_type':model_type}
    
            
    def learning_based_method(self, net, word, z, top_n):
        '''
            Parameters:
                net (chainer.Chain) ... model
                word (str) ... input word for the model
                z (int) ... input attribute ID for the model
                top_n (int) ... Number of top-N similar words to return nearest_words
        '''
        try:
            x = self.word2vec[word].astype(np.float32)
        except KeyError:
            # the word not in vocabulary
            if word[0]=='(' and word[-1]==')': # (oov_word) -> oov_word
                word = word[1:-1]
            return [('({})'.format(word), 'n/a') for _ in range(top_n)]
            
        # To variable
        x = Variable(x.reshape((1,len(x))))
        z = Variable(np.array([[z]]).astype(np.int32)) 
        #z = Variable(z.reshape(1,len(z)))
    
        # Transform a word attribute z from x into y with reflection
        y = net.test(x, z)
        
        # Show top five similar words to y
        y = y.array[0]

        nearest_words = self.word2vec.similar_by_vector(y, top_n) #self.word2vec.most_similar([y], [], 5)
        
        nearest_words = [(word[0], round(word[1], 4)) for word in nearest_words]
        return nearest_words
    
    
    def analogy_based_method(self, net, word, z, top_n):
        try:
            x = self.word2vec[word]
        except KeyError:
            # the word not in vocabulary
            if word[0]=='(' and word[-1]==')': # (oov_word) -> oov_word
                word = word[1:-1]
            return [('({})'.format(word), 'n/a') for _ in range(top_n)]
            
        # Transform a word attribute z from x into y with reflection
        y = net.test(x, self.word2vec)
        
        # Show top five similar words to y
        nearest_words = self.word2vec.similar_by_vector(y, top_n) #self.word2vec.most_similar([y], [], 5)
        
        nearest_words = [(word[0], round(word[1], 4)) for word in nearest_words]
        return nearest_words
    
    
    def get_nearest(self, net, word, z_id, top_n):
        # To variable
        nearest_words = self.word2vec.similar_by_word(word, top_n)
        nearest_words = [(word[0], round(word[1], 4)) for word in nearest_words]
        return nearest_words
    
    
    def transform_attributes(self, net, z_id, sentence, setting, show):
        #input_wtypes = [get_wordtype(w, train, val, test) for w in sentence]
        if z_id==-2:
            f = self.get_nearest
        elif net.net_name=='analogy':
            f = self.analogy_based_method
        else:
            f = self.learning_based_method
        
        # Transform and get nearest words
        nearest_ys = []
        for word in sentence:
            nearests = f(net, word, z_id, top_n=1)
            nearests = [(n[0], n[1]) for n in nearests] #n[0]: output of the transfer,  n[1]: cos sim
            nearest_ys.append(nearests)
    
        # Print transformed words
        out_words = ''
        for nearest in nearest_ys:
            out_words += nearest[0][0] + ' '
        out_words = out_words.strip()
        if show:
            print('Output:\t{}'.format(out_words))        
        
        if show:
            print('------------------------------------------------------------------------------\n')
        return out_words
     
        
    def __call__(self, sentence, z, net_setting, show=True):    
        net = net_setting['net']
        setting = net_setting['setting']
        model_type = net_setting['model_type']
        # Transform
        try:
            if show:
                print('Input:\t{}'.format(sentence))
            sentence = sentence.strip().split(' ')
            # Transform
            return self.transform_attributes(net, z, sentence, setting, show)
        except KeyError as e:
            print(e)
    
