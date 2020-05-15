import numpy
import chainer
from chainer import Variable
from tqdm import tqdm
from statistics import mean
import random
import data_loader
import copy
import json


def evaluation1_acc(xs, ts, zs, net, word2vec, n_top, dtype):
    results = get_accuracy(xs, ts, zs, net, word2vec, n_top, show=False) # n_top = [1,2,3]
    for n in n_top:
        acc = results['mean_acc@{}'.format(n)]
        print('{}@{}: {}'.format(dtype, n, acc))
    return results
    
    
def evaluation1_stb(net, z_id, word2vec, M, F, num_N, n_top):    
    results = get_stability_score(net, word2vec, M, F, z_id, num_N, n_top, show=False)
    for n in n_top:
        stb = results['mean_stb@{}'.format(n)]
        print('stability@{}: {}'.format(n, stb))
    return results
    

def get_accuracy(xs, ts, zs, net, word2vec, n_top=[1,2,3], show=False):
    results = {}
    for n in n_top:
        results['acc@{}'.format(n)] = []
    
    results['x'] = []
    results['z'] = []
    results['t'] = []
    results['nearest_words'] = []
    for x, t, z in tqdm(zip(xs, ts, zs)):
        nearest_words = _get_nearest_words(net, word2vec, x, z, n_top, show) # zはID. e.g. 0  
        results['x'].append(x)
        results['z'].append(int(z))
        results['t'].append(t)
        results['nearest_words'].append(nearest_words)
        # Accuracy
        for n in n_top:
            if t in [w[0] for w in nearest_words[:n]]:
                results['acc@{}'.format(n)].append(1)
            else:
                results['acc@{}'.format(n)].append(0)
                
    for n in n_top:
        acc = results['acc@{}'.format(n)]
        results['mean_acc@{}'.format(n)] = mean(acc)
    
    return results


def _get_nearest_words(net, word2vec, word, z_id, n_top, show=True):
    # To variable
    x = word2vec[word].astype(numpy.float32)
    x = Variable( x.reshape((1,len(x))) )
    #z = Variable(z.reshape(1,len(z)))
    z = Variable( numpy.array([[z_id]]).astype(numpy.int32) )

    # Transform a word attribute z from x into y with reflection
    y = net.test(x, z)

    # Show top five similar words to y
    y = y.array[0]
    n_nearest = max(n_top)
    nearest_words = word2vec.similar_by_vector(y, topn=n_nearest) #word2vec.most_similar([y], [], n_nearest)
    
    if show:
        print(word, nearest_words)
    
    return nearest_words


def get_stability_score(net, word2vec, M, F, z_id, num_N=1000, n_top=[1,2,3], show=False):
    vocab = list(word2vec.vocab.keys())
    vocab.sort() 
    N_test = data_loader.get_non_attribute_words(num_N, M, F, vocab)
    stability = get_stability(N_test, z_id, net, word2vec, n_top, show)
    return stability
    

def get_stability(non_attribute_words, z_id, net, word2vec, n_top, show=False):
    results = {}
    for n in n_top:
        results['stb@{}'.format(n)] = []
    
    results['x'] = []
    results['z'] = []
    results['t'] = []
    results['nearest_words'] = []
    for x in tqdm(non_attribute_words):
        nearest_words = _get_nearest_words(net, word2vec, x, z_id, n_top, show)
        results['x'].append(x)
        results['z'].append(int(z_id))
        results['t'].append(x)
        results['nearest_words'].append(nearest_words)
        # Stability
        for n in n_top:
            if x in [w[0] for w in nearest_words[:n]]: 
                results['stb@{}'.format(n)].append(1)
            else:
                results['stb@{}'.format(n)].append(0)
                if show:
                    print(x, nearest_words)                
    for n in n_top:
        stb = results['stb@{}'.format(n)]
        results['mean_stb@{}'.format(n)] = mean(stb)
    
    return results


def get_mirror_distance(v, a, c):
    # A distance between v and a mirror
    return np.square(np.dot(v-c, a)) / np.dot(a, a) # scalor


def get_xy_mirror_distance(net, word2vec, word, z_id):
    # To variable
    x = word2vec[word].astype(numpy.float32)
    x = Variable( x.reshape((1,len(x))) )
    #z = Variable(z.reshape(1,len(z)))
    z = Variable( numpy.array([[z_id]]).astype(numpy.int32) )

    # Transform a word attribute z from x into y with reflection
    y = net.test(x, z).array[0]
    x = x.array[0]
    a = net.embed_a(net.z1(z), x).array[0]
    c = net.embed_c(net.z1(z), x).array[0]

    # Calulate the distance between x/y and a mirror
    xd = get_mirror_distance(x, a, c)
    yd = get_mirror_distance(y, a, c)
    
    return xd, yd