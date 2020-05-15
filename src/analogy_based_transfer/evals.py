import numpy
from tqdm import tqdm
from statistics import mean
import random
import copy
import json
import data


def analogy(x, diff_vec, plus_minus):
    '''
        x: ndarray
        diff_vec: ndarray
        plus_minus: '+' or '-'
    '''
    if plus_minus=='+':
        return x + diff_vec
    elif plus_minus=='-':
        return x - diff_vec
    else:
        raise ValueError('{} ... No such operation'.format(plus_minus))
        
        
        

def get_mean_score_of_mean_one_diff_score(result_list, eval_method, dtype, n_top):
    results = {}
    for n in n_top:
        accs = []
        for r in result_list:
            if eval_method == 'accuracy':
                score = 'mean_acc@{}'.format(n)
            elif eval_method == 'stability':
                score = 'mean_stb@{}'.format(n)
            one_diff_vec_mean_acc = r[eval_method][dtype][score] 
            accs.append(one_diff_vec_mean_acc) 
        results['mean_acc@{}'.format(n)] = mean(accs)
        print('{}@{}: {}'.format(dtype, n, results['mean_acc@{}'.format(n)]))
    return results



def evaluation1_acc(xs, ts, zs, diff_vec, plus_minus_list, word2vec, n_top, dtype):
    results = get_accuracy(xs, ts, zs, diff_vec, plus_minus_list, word2vec, n_top, show=False) # n_top = [1,2,3]
    for n in n_top:
        acc = results['mean_acc@{}'.format(n)]
        print('{}@{}: {}'.format(dtype, n, acc))
    return results
    
    
def evaluation1_stb(diff_vec, plus_minus_list, z_id, word2vec, M, F, stb_sampling, n_top):    
    results = get_stability_with_vocab(diff_vec, plus_minus_list, word2vec, M, F, z_id, stb_sampling, n_top, show=False)
    for n in n_top:
        stb = results['mean_stb@{}'.format(n)]
        print('stability@{}: {}'.format(n, stb))
    return results
    

def get_accuracy(xs, ts, zs, diff_vec, plus_minus_list, word2vec, n_top=[1,2,3], show=False):
    results = {}
    for n in n_top:
        results['acc@{}'.format(n)] = []
    
    results['x'] = []
    results['z'] = []
    results['t'] = []
    results['nearest_words'] = []
    for x, t, z, plus_minus in tqdm(zip(xs, ts, zs, plus_minus_list)):
        nearest_words = _get_nearest_words(diff_vec, plus_minus, word2vec, x, n_top, show) # zはID. e.g. 0  
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


def _get_nearest_words(diff_vec, plus_minus, word2vec, word, n_top, show=True):
    # To variable
    x = word2vec[word].astype(numpy.float32)
    #x = Variable( x.reshape((1,len(x))) )
    #z = Variable(z.reshape(1,len(z)))
    #z = Variable( numpy.array([[z_id]]).astype(numpy.int32) )

    # Transform a word attribute z from x into y with analogy
    y = analogy(x, diff_vec, plus_minus)

    # Show top five similar words to y
    #y = y.array[0]
    n_nearest = max(n_top)
    #nearest_words = word2vec.most_similar([y], [], n_nearest)
    nearest_words = word2vec.similar_by_vector(y, topn=n_nearest) #word2vec.most_similar([y], [], n_nearest)
    
    if show:
        print(word, nearest_words)
    
    return nearest_words




def get_stability_with_vocab(diff_vec, plus_minus_list, word2vec, M, F, z_id, stb_sampling=1000, n_top=[1,2,3], show=False):
    vocab = list(word2vec.vocab.keys())
    vocab.sort() 
    N_test = data.get_N(stb_sampling, M, F, vocab)
    stability = get_stability(N_test, z_id, diff_vec, plus_minus_list, word2vec, n_top, show)
    return stability

    
def get_stability(x_samples, z_id, diff_vec, plus_minus_list, word2vec, n_top, show=False):
    results = {}
    for n in n_top:
        results['stb@{}'.format(n)] = []
    
    results['x'] = []
    results['z'] = []
    results['t'] = []
    results['nearest_words'] = []
    for x, plus_minus in tqdm(zip(x_samples, plus_minus_list)):
        nearest_words = _get_nearest_words(diff_vec, plus_minus, word2vec, x, n_top, show) 
        results['x'].append(x)
        results['z'].append(int(z_id))
        results['t'].append(x)
        results['nearest_words'].append(nearest_words)
        # Accuracy
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


    