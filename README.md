[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE.txt)

Reflection-based Word Attribute Transfer
====

## Requirements
- python  >= 3.5
- Pytorch >= 1.5.1
- gensim  >= 3.7.3
- numpy   >= 1.6.1
- tqdm
- nltk

## Setup
- Install packages according to the requirements
- Download pre-trained [word2vec(GoogleNews-vectors-negative300.bin)(3.6GB)](https://code.google.com/archive/p/word2vec/) and move it to ```data/```
```
$ cd reflection/data
$ wget -c https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
$ gunzip GoogleNews-vectors-negative300.bin.gz
```

- Download pre-trained [GloVe(glove.42B.300d.txt)(5.0GB)]( https://nlp.stanford.edu/projects/glove/) and move it to ```data```
- Fix GloVe file to read it with gensim
```
$ cd reflection/data
$ wget http://nlp.stanford.edu/data/glove.42B.300d.zip
$ unzip glove.42B.300d.zip
$ cd reflection/src
$ python fix_glove_file.py
``` 

## Usage
#### Training
``` 
$ python train.py --attr MF
``` 
#### Transfer test
``` 
$ python trans.py --attr MF --model-dir ./result/model 
``` 


## Citing
```
@inproceedings{ishibashi-etal-2020-reflection,
    title = "Reflection-based Word Attribute Transfer",
    author = "Ishibashi, Yoichi  and
              Sudoh, Katsuhito  and
              Yoshino, Koichiro  and
              Nakamura, Satoshi",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-srw.8",
    pages = "51--58"
}
```
