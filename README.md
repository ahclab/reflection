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
- Download pre-trained [GloVe(glove.42B.300d.txt)(5.0GB)]( https://nlp.stanford.edu/projects/glove/) and move it to ```data/glove/```
- Fix GloVe file to read it with gensim
 
 ```
 $ fix_glove_file.py
 ``` 

## Usage
#### Demo
- Open and run ``demos/demo.ipynb``.

#### Experiment
- Open and run ``notebooks/Ref+PM.ipynb``.
    
