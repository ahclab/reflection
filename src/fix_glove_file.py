from gensim.scripts.glove2word2vec import glove2word2vec
path_old = '../data/glove.42B.300d.txt' 
path_new = '../data/glove.42B.300d_gensim.txt' 
glove2word2vec(path_old, path_new)
