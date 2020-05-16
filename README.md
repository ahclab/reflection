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

## Directory
```
word_attribute_transfer
  ├── data                             - Datasets
  ├── demos                            - A demo notebook and trained models
  ├── notebooks                        - Experimental notebooks
  ├── src                              - Codes of word attribute transfers
  │     ├── analogy_based_transfer     - Analogy-based word attribute transfer
  │     ├── reflection_based_transfer  - Learning-based (Reflection and MLP) word attribute transfer
  │     └── fix_glove_file.py          - Script to fix GloVe file
  ├── LICENCE.txt
  └── README.md
```

## Setup
- Install packages according to the requirements
- Download pre-trained [word2vec(GoogleNews-vectors-negative300.bin)](https://code.google.com/archive/p/word2vec/) and move it to ```data/word2vec/```
- Download pre-trained [GloVe(glove.42B.300d.txt)]( https://nlp.stanford.edu/projects/glove/) and move it to ```data/glove/```
- Fix GloVe file to read it with gensim
 
 ```
 $ fix_glove_file.py
 ``` 

## Usage
#### Demo
- Open and run ``demos/demo.ipynb``.

#### Experiment
- Open and run ``notebooks/Ref+PM.ipynb``.
    
