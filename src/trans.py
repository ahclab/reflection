import numpy as np
import argparse
import json
import os
import nltk
from tqdm import tqdm
import torch
import utils
from nets import Ref_PM, Ref_PM_Share

ATTR2ID = {'MF':0, 'SP':1, 'CC':2, 'AN':3}

def predict(model, device, x_batch, z_batch):
    model.eval()
    with torch.no_grad():
        x_batch, z_batch =  x_batch.to(device), z_batch.to(device)
        output = model(x_batch, z_batch)
    return output.to('cpu').detach().numpy()


def trasfer_from_tokens(model, device, tokens, z, word_embedding, demo_mode=False):
    '''
        Examples:
        >>> device = torch.device("cpu")
        >>> model = Ref_PM(300, 300).to(device)
        >>> model.load_state_dict(torch.load(model_path, map_location=device))
        >>> tokens = [["he", "is", "a", "boy", "."], 
                      ["she", "is", "a", "heroine", "."]]
        >>> z = 0 # transfer target: gender
        >>> trasfer_from_tokens(model, device, tokens, z, word_embedding)
             [["she", "is", "a", "girl", "."], 
              ["he", "is", "a", "hero", "."]]
    '''
    z = torch.from_numpy(np.array([z]))
    result = []
    if not demo_mode:
        tokens = tqdm(tokens)
    for sentence in tokens:
        oov_word = []
        x = []
        for i, word in enumerate(sentence):
            try:
                x.append(word_embedding[word])
            except:
                oov_word.append((i, word))
        if x==[]:
            result.append(sentence)  # result becomes the input if oov tokens only
        else:
            x = torch.from_numpy(np.array(x)).float()
            output_vectors = predict(model, device, x, z)
            output_words = []
            for y in output_vectors:
                output_words.append(word_embedding.similar_by_vector(y, 1)[0][0])
            for i, oov in enumerate(oov_word):
                output_words.insert(oov[0], oov[1])
            result.append(output_words)
    return result


def main():
    parser = argparse.ArgumentParser(description='PyTorch Reflection-based Word Attribute Transfer Example')
    parser.add_argument('--model-dir', type=str, 
                        help='model directory')
    parser.add_argument('--attr', type=str, 
                        help='target attribute to transfer {MF, SP, CC, AN}')
    parser.add_argument('--src', type=str, default='',
                        help='path of source file. demo mode if no value is set')
    parser.add_argument('--no-tokenize', action='store_true', default=False,
                        help='disables tokenization (default: False)')
    parser.add_argument('--gpu', type=int, default=-1, 
                        help='gpu device id. -1 indicates cpu (default: -1)')

    args = parser.parse_args()
    with open(args.model_dir + '/args.json') as f:
        config = json.load(f)
        config.update(args.__dict__)
        args.__dict__ = config
    print(json.dumps(args.__dict__, indent=2))
        
    torch.manual_seed(args.seed)
    device, use_cuda = utils.get_device(args.gpu)
        
    # Load model
    print('loading model...')
    if args.weight_sharing:
        model = Ref_PM_Share(args.dim_x, args.dim_h).to(device)
    else:
        model = Ref_PM(args.dim_x, args.dim_h).to(device)
    model_path = args.model_dir + '/model.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('loaded.')
        
    # Load word embeddings
    print('loading word embeddings...')
    word_embedding = utils.load_word_embeddings(args.emb)
    print('loaded.')

    # Transfer
    z = ATTR2ID[args.attr]
    demo_mode = True if not args.src else False
    if demo_mode:
        print('\n[demo mode]')
        while(1):
            sentence = input('input:  ')
            tokens = [nltk.word_tokenize(sentence)]
            result = trasfer_from_tokens(model, device, tokens, z, word_embedding, demo_mode)
            print('output: ' + ' '.join(result[0]))
    else:
        # Transfer text file
        with open(args.src) as f:
            src = f.read().split('\n')
        if args.emb=='glove':
            src = list(map(str.lower, src))
        if args.no_tokenize:
            tokens = [sentence.split(' ') for sentence in src]
        else:
            tokens = [nltk.word_tokenize(sentence) for sentence in src]
        result = trasfer_from_tokens(model, device, tokens, z, word_embedding, demo_mode)
    
        # Save result
        filename = args.src.split('/')[-1].split('.')[0]
        path = args.model_dir + '/result_' + args.attr + '_' + filename + '.txt'
        with open(path, 'w') as f:
            r = '\n'.join([' '.join(tokens) for tokens in result])
            f.write(r)
        print('results are saved at ', path)
        
if __name__ == '__main__':
    '''
        Example:
            $ python trans.py --attr MF --model-dir ./result/[MODEL DIRECTORY] # domo mode
            $ python trans.py --attr MF --model-dir ./result/[MODEL DIRECTORY] --src sentence_example.txt
    '''
    main()