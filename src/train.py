import numpy as np
import argparse
import json
import os
import datetime
from statistics import mean
from tqdm import tqdm
import torch
#from torch.optim.lr_scheduler import StepLR
import utils
import eval
from nets import Ref_PM, Ref_PM_Share

'''
    Attributes:
        MF: male-female
        SP: singular-plural
        CC: capital-country
        AN: antonym
'''
ATTR2ID = {'MF':0, 'SP':1, 'CC':2, 'AN':3}


def add_noise_to_word_vec(wv, sigma):
    return wv + torch.from_numpy(sigma * np.random.randn(*wv.shape)).float()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    mse = torch.nn.MSELoss()
    train_loss = 0
    for batch_idx, (x, target, z) in enumerate(train_loader):
        x = add_noise_to_word_vec(x, args.sigma) # add noise
        x, z, target =  x.to(device), z.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x, z)
        loss = mse(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        if args.dry_run:
            break
    return train_loss / len(train_loader.dataset)


def test(model, device, valid_loader, dtype):
    model.eval()
    mse = torch.nn.MSELoss() #reduction='sum')
    test_loss = 0
    with torch.no_grad():
        for x, target, z in valid_loader:
            x, z, target =  x.to(device), z.to(device), target.to(device)
            output = model(x, z)
            test_loss += mse(output, target) # sum up batch loss
    return test_loss / len(valid_loader.dataset)


def get_valid_score(model, device, valid_data, word_embedding):
    X_valid, Z_valid, T_valid = valid_data
    Y_valid = eval.trasfer(model, device, X_valid, Z_valid, word_embedding)
    accuracy = []
    for y, t in zip(Y_valid, T_valid):
        if type(t)==list:
            if y in t:
                accuracy.append(1)
            else:
                accuracy.append(0)
        else:
            if y==t:
                accuracy.append(1)
            else:
                accuracy.append(0)
    return mean(accuracy)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Reflection-based Word Attribute Transfer Example')
    parser.add_argument('--emb', type=str, default='glove', 
                        help='word embeddings: word2vec or glove (default: glove)')
    parser.add_argument('--attr', type=str, default='joint', 
                        help='attribute {MF, SP, CC, AN, joint} (default: joint)')
    parser.add_argument('--use-all-data', action='store_true', default=False,
                        help='Use all data for final training (default: False)')
    parser.add_argument('--weight-sharing', action='store_true', default=True,
                        help='use weight sharing (default: False)')
    parser.add_argument('--invariant', type=float, default=0.9, 
                        help='rate of invariant words for training (default: 0.9)')
    parser.add_argument('--batch-size', type=int, default=4096, 
                        help='input batch size for training (default: 4096)')
    parser.add_argument('--valid-batch-size', type=int, default=1000, 
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, 
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate (default: 0.0001)')
    #parser.add_argument('--gamma', type=float, default=1.0, 
    #                    help='Learning rate step gamma (default: 1.0)')
    parser.add_argument('--dim-x', type=int, default=300, 
                        help='dim of word vectors (default: 300)')
    parser.add_argument('--dim-h', type=int, default=3000, 
                        help='dim of hidden units (default: 3000)')
    #parser.add_argument('--n-layers', type=int, default=5, 
    #                    help='num of layers (default: 5)')
    parser.add_argument('--sigma', type=float, default=0.1, 
                        help='gaussian noise standard deviation (default: 0.1)')
    parser.add_argument('--gpu', type=int, default=-1, 
                        help='gpu device id. -1 indicates cpu (default: -1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=250, 
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--fout', type=str, default='./result', 
                        help='output directory (default: ./result)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-dir', type=str, default='', 
                        help='directory of model for retrain (default: )')
    parser.add_argument('--invariant_word_type', default="mix", choices=["noisy", "frequent", "mix"], 
                        help='invariant word type (default: mix)')
    parser.add_argument('--valid_by_acc', action='store_true', default=False,
                        help='validation by accuracy (default: False)')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
        
    torch.manual_seed(args.seed)
    device, use_cuda = utils.get_device(args.gpu)

    # Load dataset
    train_kwargs = {'batch_size': args.batch_size}
    valid_kwargs = {'batch_size': args.valid_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        valid_kwargs.update(cuda_kwargs)
        
    # Load word embeddings
    print('loading word embeddings...')
    word_embedding = utils.load_word_embeddings(args.emb)
    print('loaded.')
    
    assert args.attr in ['MF', 'SP', 'CC', 'AN', 'joint']
    if args.attr=='joint':
        attributes = ['MF', 'SP', 'CC', 'AN']
    else:
        attributes = [args.attr]
    
    # Create train/val/test data
    if args.use_all_data:
        print('loading datasets...')
        dataset = utils.load_dataset(args.invariant, attributes, args.seed, args.emb, 
                                     invariant_word_type=args.invariant_word_type)
        print('loaded.')
        print('creating data loader')
        X_train = torch.from_numpy(np.array([word_embedding[d[3]] for d in dataset])).float()
        T_train = torch.from_numpy(np.array([word_embedding[d[4]] for d in dataset])).float()
        Z_train = torch.from_numpy(np.array([ATTR2ID[d[2]] for d in dataset]))
        dataset1 = torch.utils.data.TensorDataset(X_train, T_train, Z_train)    
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        print('created.')
    else:
        include_one_to_many_data = True if args.valid_by_acc else False
        print('loading datasets...')
        dataset = utils.load_dataset(args.invariant, attributes, args.seed, args.emb, 
                                     include_one_to_many_data=include_one_to_many_data,
                                     invariant_word_type=args.invariant_word_type)
        print('loaded.')
        
        X_train = [d[3] for d in dataset if d[1]=='train']
        X_valid = [d[3] for d in dataset if d[1]=='valid']
        X_test = [d[3] for d in dataset if d[1]=='test']
        T_train = [d[4] for d in dataset if d[1]=='train']
        T_valid = [d[4] for d in dataset if d[1]=='valid']
        T_test = [d[4] for d in dataset if d[1]=='test']
        Z_train = [ATTR2ID[d[2]] for d in dataset if d[1]=='train']
        Z_valid = [ATTR2ID[d[2]] for d in dataset if d[1]=='valid']
        Z_test = [ATTR2ID[d[2]] for d in dataset if d[1]=='test']
    
        print('creating data loader')
        X_train = torch.from_numpy(np.array([word_embedding[w] for w in X_train])).float()
        T_train = torch.from_numpy(np.array([word_embedding[w] for w in T_train])).float()
        Z_train = torch.from_numpy(np.array(Z_train))
        dataset1 = torch.utils.data.TensorDataset(X_train, T_train, Z_train)        
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        
        if not args.valid_by_acc:
            X_valid = torch.from_numpy(np.array([word_embedding[w] for w in X_valid])).float()
            T_valid = torch.from_numpy(np.array([word_embedding[w] for w in T_valid])).float()
            Z_valid = torch.from_numpy(np.array(Z_valid))
            X_test = torch.from_numpy(np.array([word_embedding[w] for w in X_test])).float()
            T_test = torch.from_numpy(np.array([word_embedding[w] for w in T_test])).float()
            Z_test = torch.from_numpy(np.array(Z_test))
            dataset2 = torch.utils.data.TensorDataset(X_valid, T_valid, Z_valid)
            dataset3 = torch.utils.data.TensorDataset(X_test, T_test, Z_test)
            valid_loader = torch.utils.data.DataLoader(dataset2, **valid_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset3, **valid_kwargs)
        else:
            valid_data = (X_valid, Z_valid, T_valid)
            test_data = (X_test, Z_test, T_test)
        print('created.')
    
    # Set model
    if args.weight_sharing:
        model = Ref_PM_Share(args.dim_x, args.dim_h).to(device)
        args.model = 'Ref+PM+Share'
    else:
        model = Ref_PM(args.dim_x, args.dim_h).to(device)
        args.model = 'Ref+PM'
    if args.model_dir:
        model_path = args.model_dir + '/model.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('model loaded.')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Make dir
    now = str(datetime.datetime.today()).replace(' ', '_').replace(':', '-').replace('.', '_')
    experiment_name = '{}_{}_{}_{}'.format(args.emb, args.attr, args.model, now)
    args.exp_name = experiment_name
    path_save = '{}/{}'.format(args.fout, experiment_name)
    os.makedirs(path_save, exist_ok=True)
    with open('{}/args.json'.format(path_save), 'w') as f:
        json.dump(args.__dict__, f)
    
    if args.dry_run:
        args.epochs = 1
    
    # Training
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_score, best_loss = -1, 1e+10
    save_flag = False
    training_loop = tqdm(range(1, args.epochs + 1))
    for epoch in training_loop:
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        if args.use_all_data:
            training_loop.set_description("Train Epoch %d | Train Loss: %f" % (epoch, train_loss))
        elif epoch % args.log_interval == 0:
            if args.valid_by_acc:
                valid_score = get_valid_score(model, device, valid_data, word_embedding)
                training_loop.set_description("Train Epoch %d | Train Loss: %f | Valid Score: %f" % (epoch, train_loss, valid_score))
                if valid_score > best_score:
                    save_flag = True
                    best_score = valid_score
                else:
                    save_flag = False
            else:
                valid_loss = test(model, device, valid_loader, 'Valid')
                training_loop.set_description("Train Epoch %d | Train Loss: %f | Valid Loss: %f" % (epoch, train_loss, valid_loss))
                if valid_loss < best_loss:
                    save_flag = True
                    best_loss = valid_loss
                else:
                    save_flag = False
        #scheduler.step()
        
        # Logging training status
        if args.save_model and epoch % args.log_interval == 0:
            if args.use_all_data or save_flag:
                torch.save(model.state_dict(), "{}/model.pt".format(path_save))
    # Test
    model_path = "{}/model.pt".format(path_save)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if not args.use_all_data:
        if args.valid_by_acc:
            test_score = get_valid_score(model, device, valid_data, word_embedding)
            print('Test Score: {:.6f}'.format(test_score))
        else:
            test_loss = test(model, device, test_loader, 'Test')
            print('Test Loss: {:.6f}'.format(test_loss))
        
if __name__ == '__main__':
    '''
        Example:
            $ python train.py --save-model --attr joint
    '''
    main()
