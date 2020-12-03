'''
    Step1. Download pre-trained GloVe(glove.42B.300d.txt) 
    from( https://nlp.stanford.edu/projects/glove/) 
    and put it in data/

    Step2. Run this script.
'''

from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == '__main__':
    path_old = '../data/glove.42B.300d.txt' 
    path_new = '../data/glove.42B.300d_gensim.txt' 

    _ = glove2word2vec(path_old, path_new)
