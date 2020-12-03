import numpy as np

import chainer
from chainer import Chain, Variable
from chainer import functions as F
from chainer import links as L


def reflection_numpy(x, a, c):
    '''
        e.g., 
            x = np.array([1,1])
            a = np.array([1,1])
            c = np.array([0,0])
            return np.array([-1,-1])
    '''
    return x - 2 * (np.dot(x-c, a) / np.dot(a, a)) * a


def reflection(x, a, c):
    assert x.shape==a.shape==c.shape, \
    'x.shape={}, a.shape={}, c.shape={}'.format(x.shape, a.shape, c.shape)
    # Reflection
    dot_aa = F.batch_matmul(a, a, transa=True) 
    dot_aa = F.reshape(dot_aa, (a.shape[0],1))
    dot_xca = F.batch_matmul(x-c, a, transa=True) 
    dot_xca = F.reshape(dot_xca, (c.shape[0],1))
    return x - 2 * ( dot_xca / dot_aa) * a


def add_noise_to_word_vec(wv, sigma=.1):
    if chainer.config.train:
        xp = chainer.backend.get_array_module(wv.array)
        return wv + sigma * xp.random.randn(*wv.shape)
    else:
        return wv


    
class Ref(chainer.Chain): 
    ''' Reflection-based word attribute transfer with a single mirror'''
    def __init__(self, dim_x, dim_h, n_attribute, sigma):
        '''
        Args
             dim_x: dim of word2vec/GloVe
             dim_h: dim of MLP hidden layers
             n_attribute: num of target attribute
             sigma: standard deviation of Gaussian distribution (Gaussian noise)
        '''
        super(Ref, self).__init__()
        self.sigma = sigma
        self.net_name = 'Ref'

        with self.init_scope():
            self.wa1 = L.Linear(dim_x, dim_x) # self.wa1 = L.Linear(dim_x, dim_h)
            #self.wa2 = L.Linear(dim_h, dim_h)
            #self.wa3 = L.Linear(dim_h, dim_x)
            self.wc1 = L.Linear(dim_x, dim_x) # self.wc1 = L.Linear(dim_x, dim_h)
            #self.wc2 = L.Linear(dim_h, dim_h)
            #self.wc3 = L.Linear(dim_h, dim_x)
            self.embed_z = L.EmbedID(n_attribute, dim_x)
    
    def mlp_a(self, z):
        a = self.wa1(z)
        #a = self.wa3(F.relu(a))
        #a = self.wa4(F.relu(a))
        return a

    def mlp_c(self, z):
        c = self.wc1(z)
        #c = self.wc3(F.relu(c))
        #c = self.wc4(F.relu(c))
        return c

    def forward(self, x, z):   
        '''
        Args
           x: input word vector
           z: attribute ID (e.g. 0)
        Return:
           y: output vector
        '''
        # Add noise
        x = add_noise_to_word_vec(x, self.sigma)
        
        # Embed z
        z = self.embed_z(z)
        
        # Estimate a
        a = self.mlp_a(z) # Single mirror
        
        # Estimate c
        c = self.mlp_c(z) # Single mirror
        
        # Transfer the word vector with reflection
        y = reflection(x, a, c) # y = Ref_a,c(x)
        
        return y
        
    def loss(self, x, t, z):
        '''
        Args
           x: input word vector
           z: attribute ID (e.g. 0)
           t: target word vector
        Return:
           mean squared error between y and t
        '''
        y = self.forward(x, z)
        return F.mean_squared_error(y, t)
    
    def test(self, x, z):
        with chainer.using_config('train', False):
            y = self.forward(x, z)
        return y
    
    
    
    
class Ref_PM(chainer.Chain): 
    ''' Reflection-based word attribute transfer with parameterized mirrors'''
    def __init__(self, dim_x, dim_h, n_attribute, sigma):
        '''
        Args
             dim_x: dim of word2vec/GloVe
             dim_h: dim of MLP hidden layers
             n_attribute: num of target attribute
             sigma: standard deviation of Gaussian distribution (Gaussian noise)
        '''
        super(Ref_PM, self).__init__()
        self.sigma = sigma
        self.net_name = 'Ref+PM'

        with self.init_scope():
            self.wa1 = L.Linear(dim_x, dim_h)
            self.wa2 = L.Linear(dim_x, dim_h)
            self.wa3 = L.Linear(dim_h, dim_h)
            self.wa4 = L.Linear(dim_h, dim_x)
            self.wc1 = L.Linear(dim_x, dim_h)
            self.wc2 = L.Linear(dim_x, dim_h)
            self.wc3 = L.Linear(dim_h, dim_h)
            self.wc4 = L.Linear(dim_h, dim_x)
            self.embed_z = L.EmbedID(n_attribute, dim_x)
    
    def mlp_a(self, z, x):
        a = self.wa1(z) + self.wa2(x)
        a = self.wa3(F.relu(a))
        a = self.wa4(F.relu(a))
        return a

    def mlp_c(self, z, x):
        c = self.wc1(z) + self.wc2(x)
        c = self.wc3(F.relu(c))
        c = self.wc4(F.relu(c))
        return c

    def forward(self, x, z):   
        '''
        Args
           x: input word vector
           z: attribute ID (e.g. 0)
        Return:
           y: output vector
        '''
        # Add noise
        x = add_noise_to_word_vec(x, self.sigma)
        
        # Embed z
        z = self.embed_z(z)
        
        # Estimate a
        a = self.mlp_a(z, x) # Parameterized mirror
        
        # Estimate c
        c = self.mlp_c(z, x) # Parameterized mirror
        
        # Transfer the word vector with reflection
        y = reflection(x, a, c) # y = Ref_a,c(x)
        
        return y
        
    def loss(self, x, t, z):
        '''
        Args
           x: input word vector
           z: attribute ID (e.g. 0)
           t: target word vector
        Return:
           mean squared error between y and t
        '''
        y = self.forward(x, z)
        return F.mean_squared_error(y, t)
    
    def test(self, x, z):
        with chainer.using_config('train', False):
            y = self.forward(x, z)
        return y
    


class MLP2(chainer.Chain): 
    
    def __init__(self, dim_x, dim_h, n_attribute, sigma, drop_ratio):
        super(MLP2, self).__init__()
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.n_attribute = n_attribute
        self.sigma = sigma
        self.drop_ratio = drop_ratio
        self.net_name = 'MLP2'
        
        with self.init_scope():
            self.wx = L.Linear(dim_x, dim_h) # to concat [x;z]
            self.wz = L.Linear(dim_x, dim_h) # to concat [x;z]
            self.w1 = L.Linear(dim_h, dim_h)
            self.w2 = L.Linear(dim_h, dim_x)
            self.emb_z = L.EmbedID(n_attribute, dim_x)    

    
    def mlp(self, x, z):
        # Concat x and z
        z = self.wz(z)
        x = self.wx(x)
        x = x + z
        # MLP
        y = F.dropout(F.relu(x), self.drop_ratio)
        y = self.w1(y)
        y = F.dropout(F.relu(y), self.drop_ratio)
        y = self.w2(y)
        return y
        
    def forward(self, x, z):    
        # Add noise
        x = add_noise_to_word_vec(x, self.sigma)
        
        # Embed z
        z = self.emb_z(z)
        
        # Pred y by using MLP
        y = self.mlp(x, z)
                
        return y

    def loss(self, x, t, z):
        '''
        Args
           x: input word vector
           z: attribute ID (e.g. 0)
           t: target word vector
        Return:
           mean squared error between y and t
        '''
        y = self.forward(x, z)
        return F.mean_squared_error(y, t)

    def test(self, x, z):
        with chainer.using_config('train', False):
            y = self.forward(x, z)
        return y
    

    

class MLP3(chainer.Chain): 
    
    def __init__(self, dim_x, dim_h, n_attribute, sigma, drop_ratio):
        super(MLP3, self).__init__()
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.n_attribute = n_attribute
        self.sigma = sigma
        self.drop_ratio = drop_ratio
        self.net_name = 'MLP3'
        
        with self.init_scope():
            self.wx = L.Linear(dim_x, dim_h)
            self.wz = L.Linear(dim_x, dim_h)
            self.w1 = L.Linear(dim_h, dim_h)
            self.w2 = L.Linear(dim_h, dim_h)
            self.w3 = L.Linear(dim_h, dim_x)
            self.emb_z = L.EmbedID(n_attribute, dim_x) 

    
    def mlp(self, x, z):
        # Concat x and z
        z = self.wz(z)
        x = self.wx(x)
        x = x + z
        # MLP
        y = F.dropout(F.relu(x), self.drop_ratio)
        y = self.w1(y)
        y = F.dropout(F.relu(y), self.drop_ratio)
        y = self.w2(y)
        y = F.dropout(F.relu(y), self.drop_ratio)
        y = self.w3(y)
        return y
        
    def forward(self, x, z):    
        # Add noise
        x = add_noise_to_word_vec(x, self.sigma)
        
        # Embed z
        z = self.emb_z(z)
        
        # Pred y by using MLP
        y = self.mlp(x, z)
                
        return y

    def loss(self, x, t, z):
        '''
        Args
           x: input word vector
           z: attribute ID (e.g. 0)
           t: target word vector
        Return:
           mean squared error between y and t
        '''
        y = self.forward(x, z)
        return F.mean_squared_error(y, t)

    def test(self, x, z):
        with chainer.using_config('train', False):
            y = self.forward(x, z)
        return y