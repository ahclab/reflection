import numpy as np
from statistics import mean
from tqdm import tqdm
import argparse
import json
import os
import torch
import utils
import trans
from nets import Ref_PM, Ref_PM_Share

'''
    Attributes:
        MF: male-female
        SP: singular-plural
        CC: capital-country
        AN: antonym
'''
ATTR2ID = {'MF':0, 'SP':1, 'CC':2, 'AN':3}


def trasfer(model, device, X, Z, word_embedding):
    '''
        Examples:
        >>> device = torch.device("cpu")
        >>> model = Ref_PM(300, 300).to(device)
        >>> model.load_state_dict(torch.load(model_path, map_location=device))
        >>> X = ["man", "boy", "sister", "girl", "lady"]
        >>> Z = [0, 0, 0, 0, 0] # transfer target: gender
        >>> trasfer_words(model, device, X, Z, word_embedding)
                ["man", "boy", "sister", "girl", "lady"]
    '''
    X = torch.from_numpy(np.array([word_embedding[w] for w in X])).float()
    Z = torch.from_numpy(np.array(Z))
    output_vectors = trans.predict(model, device, X, Z) # output
    output_words = [word_embedding.similar_by_vector(y, 1)[0][0] for y in tqdm(output_vectors)]
    return output_words


def main():
    parser = argparse.ArgumentParser(description='PyTorch Reflection-based Word Attribute Transfer Example')
    parser.add_argument('--model-dir', type=str, 
                        help='model directory')
    parser.add_argument('--attr', type=str, choices=["MF", "SP", "CC", "AN"],
                        help='target attribute {MF, SP, CC, AN}')
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

    # Calculate accuracy and stability
    attributes = [args.attr]
    dataset = utils.load_dataset(0, attributes, args.seed, args.emb, True)
    print('calculating accuracy...')
    #X_train = [d[3] for d in dataset if d[1]=='train' and d[0]=='A']
    #T_train = [d[4] for d in dataset if d[1]=='train' and d[0]=='A']
    #Z_train = [ATTR2ID[d[2]] for d in dataset if d[1]=='train' and d[0]=='A']
    #Y_train = trasfer(model, device, X_train, Z_train, word_embedding)
    #accuracy = mean([1 if y==t else 0 for y, t zip(Y_train, T_train)])
    #print('accuracy: %f' % accuracy)
    #X_valid = [d[3] for d in dataset if d[1]=='valid'and d[0]=='A']
    #T_valid = [d[4] for d in dataset if d[1]=='valid' and d[0]=='A']    
    #Z_valid = [ATTR2ID[d[2]] for d in dataset if d[1]=='valid' and d[0]=='A']
    #Y_valid = trasfer(model, device, X_valid, Z_valid, word_embedding)
    #accuracy = mean([1 if y==t else 0 for y, t zip(Y_valid, T_valid)])
    #print('accuracy: %f' % accuracy)
    X_test = [d[3] for d in dataset if d[1]=='test' and d[0]=='A']
    T_test = [d[4] for d in dataset if d[1]=='test' and d[0]=='A']
    Z_test = [ATTR2ID[d[2]] for d in dataset if d[1]=='test' and d[0]=='A']
    Y_test = trasfer(model, device, X_test, Z_test, word_embedding)
    accuracy = mean([1 if y==t else 0 for y, t in zip(Y_test, T_test)])
    print('accuracy: %f' % accuracy)
    
    print('calculating stability...')
    X_test = [d[3] for d in dataset if d[1]=='test' and d[0]=='N']
    T_test = [d[4] for d in dataset if d[1]=='test' and d[0]=='N']
    Z_test = [ATTR2ID[d[2]] for d in dataset if d[1]=='test' and d[0]=='N']
    Y_test = trasfer(model, device, X_test, Z_test, word_embedding)
    stability = mean([1 if y==t else 0 for y, t in zip(Y_test, T_test)])
    print('stability: %f' % stability)

if __name__ == '__main__':
    '''
        Example:
            $ python eval.py --attr MF --model-dir ./result/[MODEL DIRECTORY]
    '''
    main()