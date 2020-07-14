[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE.txt)

Reflection-based Word Attribute Transfer
====
- Demo: Demo of reflection-based word attribute transfers. 
- Experiment: You can train models by using Jupyter notebooks.

<div align="center">
<img src=./demos/demo.png "DEMO", width=500>
</div>

## Requirements
- python  >= 3.5
- chainer >= 5.0
- gensim  >= 3.7.3
- numpy   >= 1.6.1
- tqdm
- Jupyter Notebook

## Directory
```
reflection
  ├── data                             
  │     ├── datasets                   - Datasets
  │     ├── word2vec                   - Pre-trained word2vec
  │     └── glove                      - Pre-trained GloVe
  ├── demos
  │     ├── demo.ipynb                 - Demo notebook
  │     ├── models_word2vec            - Trained models
  │     └── models_glove               - Trained models
  ├── notebooks                        
  │     └── Ref+PM.ipynb               - Experimental notebook
  ├── src                              
  │     ├── analogy_based_transfer     - Codes of analogy-based word attribute transfer
  │     ├── reflection_based_transfer  - Codes of learning-based (Reflection and MLP) word attribute transfer
  │     └── fix_glove_file.py          - Script to fix GloVe file
  ├── LICENCE.txt
  └── README.md
```

## Setup
- Install packages according to the requirements
- Download pre-trained [word2vec(GoogleNews-vectors-negative300.bin)(3.6GB)](https://code.google.com/archive/p/word2vec/) and move it to ```data/word2vec/```
```
$ cd reflection/data/word2vec
$ wget -c https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
$ gunzip GoogleNews-vectors-negative300.bin.gz
```

- Download pre-trained [GloVe(glove.42B.300d.txt)(5.0GB)]( https://nlp.stanford.edu/projects/glove/) and move it to ```data/glove/```
- Fix GloVe file to read it with gensim
```
$ cd reflection/data/glove
$ wget http://nlp.stanford.edu/data/glove.42B.300d.zip
$ unzip glove.42B.300d.zip
$ cd reflection/src
$ python fix_glove_file.py
 ``` 

## Usage
#### Demo
- Open and run ``demos/demo.ipynb``.

#### Experiment
- Open and run ``notebooks/Ref+PM.ipynb``.
    
 
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
