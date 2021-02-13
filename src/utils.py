import gensim
import random
import json
import torch

path_w2v = '/home/is/yoichi-is/Research/reflection/data/word2vec/GoogleNews-vectors-negative300.bin' 
path_glv = '/home/is/yoichi-is/Research/reflection/data/glove/glove.42B.300d_gensim.txt' 
path_dataset = '/home/is/yoichi-is/Research/reflection/data/datasets_20201130' 
    

def load_word_embeddings(embedding):
    if embedding == 'word2vec':
        return gensim.models.KeyedVectors.load_word2vec_format(path_w2v, binary=True)
    elif embedding == 'glove':
        return gensim.models.KeyedVectors.load_word2vec_format(path_glv)
    

def reflection_numpy(x, a, c):
    '''
        Examples:
        >>> x = np.array([1,1])
        >>> a = np.array([1,1])
        >>> c = np.array([0,0])
        >>> reflection_numpy(x, a, c) 
            array([-1, -1])
    '''
    return x - 2 * (np.dot(x-c, a)/np.dot(a, a)) * a


def load_dataset(rate_invariant_words, attributes, seed, embedding, 
                 include_one_to_many_data=False, use_frequent_invariant_words=True):
    # Load A nad N dataset
    with open(path_dataset + '/datasets_' + embedding + '.json') as f:
        dataset = json.load(f)
    attribute_words = [d for d in dataset['A'] if d[2] in attributes]
    if use_frequent_invariant_words: # more stable type
        invariant_words = [d for d in dataset['N_frequent_words'] if d[2] in attributes]
    else:
        invariant_words = [d for d in dataset['N'] if d[2] in attributes] # paper version
    invariant_words_train = [d for d in invariant_words if d[1] == 'train']
    invariant_words_test = [d for d in invariant_words if d[1] == 'test']
    
    # Fix data
    attribute_words = many_to_one(attribute_words) if not include_one_to_many_data else attribute_words
    
    # Sampling invariant words for training
    r = rate_invariant_words
    assert 0 <= r < 1
    num_invariant_words = int(r/(1-r) * len(attribute_words))
    if num_invariant_words > len(invariant_words_train):
        num_invariant_words = len(invariant_words_train)
    random.seed(seed)
    invariant_words_train = random.sample(invariant_words_train, num_invariant_words)
    print('#A, #N_train, #N_test')
    print(len(attribute_words), len(invariant_words_train), len(invariant_words_test))
    return attribute_words + invariant_words_train + invariant_words_test
    
    
def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        print('device: gpu')
        return torch.device("cuda", gpu_id), True
    else:
        print('device: cpu')
        return torch.device("cpu"), False

    
def many_to_one(attribute_words):
    '''
        Fix data format which includes one-to-many relations 
        to be the one-to-one format
        
        Examples:
            >>> attribute_words = [(
                                    'A', 'valid', 'AN', 
                                    'good', ['bad', 'poor', 'terrible']
                                  )]
            >>> many_to_one(attribute_words)
                    [('A', 'valid', 'AN', 'good', 'bad'),
                     ('A', 'valid', 'AN', 'good', 'poor'),
                     ('A', 'valid', 'AN', 'good', 'terrible')]
    '''
    fixed_attribute_words = []
    for wtype, dtype, z, x, t in attribute_words:
        if type(t)==list:
            for _t in t:
                fixed_attribute_words.append((wtype, dtype, z, x, _t))
        else:
            fixed_attribute_words.append((wtype, dtype, z, x, t))
    return fixed_attribute_words