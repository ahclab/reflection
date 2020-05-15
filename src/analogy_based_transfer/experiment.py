import json
import os
import numpy as np
from statistics import mean

import data
import evals
import copy


def get_one_diff_vecs(word2vec, M, F):
    diff_vecs = [ word2vec[m] - word2vec[w] for m, w in zip(M, F) ]  
    pairs = [ (m, w) for m, w in zip(M, F) ] 
    return diff_vecs, pairs

def get_avg_diff_vecs(word2vec, M, F):
    diff_vecs, _ = get_one_diff_vecs(word2vec, M, F)
    return np.mean(np.asarray(diff_vecs), axis=0)
    
def get_plus_minus_list(xs_val, xs_test, stb_sampling_test, plus_minus_or_info, M, F):
    def get_plus_minus_list(xs, M, F):
        plus_minus_list = []
        for x in xs:
            if x in M: 
                plus_minus_list.append('-')
            elif x in F:
                plus_minus_list.append('+')
            else:
                assert False
        return plus_minus_list
    
    if plus_minus_or_info=='plus-all':
        plus_minus_val = ['+' for i in range(len(xs_val))]
        plus_minus_test = ['+' for i in range(len(xs_test))]
        plus_minus_stb = ['+' for i in range(stb_sampling_test)]
    elif plus_minus_or_info=='minus-all':
        plus_minus_val= ['-' for i in range(len(xs_val))]
        plus_minus_test = ['-' for i in range(len(xs_test))]  
        plus_minus_stb = ['-' for i in range(stb_sampling_test)]
    elif plus_minus_or_info=='use-knowledge':
        plus_minus_val = get_plus_minus_list(xs_val, M, F)
        plus_minus_test = get_plus_minus_list(xs_test, M, F)
        plus_minus_stb = []
    else:
        assert False
    return plus_minus_val, plus_minus_test, plus_minus_stb
        
        
def experiment1_avg_diff(settings):
    experiment_name = settings['experiment_name']
    word2vec = settings['word2vec']
    stb_sampling_train = 0 
    n_argument = 0 
    plus_minus_or_info = settings['model']['plus_minus_or_info']
    settings['model']['name'] = 'avg_diff'
    datasets = settings['datasets']
    path_dataset = settings['path_dataset'] 
    stb_sampling_test = settings['stb_sampling_test'] 
    n_top = settings['n_top'] 
    out = settings['out']
    emb_model = settings['embedding_method']
    
    # Load data
    xs_train, xs_val, xs_test, ts_train, ts_val, ts_test, zs_train, zs_val, zs_test, M, F, xs, ts, zs, A_train, A_val, A_test, z_ids, N_train = data.load(datasets, word2vec, stb_sampling_train, path_dataset, emb_model)
        
    # Get diff vector
    ms = xs_train[::2]
    ws = ts_train[::2]
    #diff_vecs = get_one_diff_vecs(word2vec, M, F)
    diff_vec = get_avg_diff_vecs(word2vec, ms, ws)
    #print(diff_vec)

    # Get plus-minus list    
    plus_minus_val, plus_minus_test, plus_minus_stb = get_plus_minus_list(xs_val, xs_test, stb_sampling_test, plus_minus_or_info, M, F)

    # Evaluation (Accuracy)
    print('\ncalculating accuracies...')
    #train_results = evals.evaluation1_acc(_xs_train, _ts_train, _zs_train, net, word2vec, n_top, 'train')
    val_results = evals.evaluation1_acc(xs_val, ts_val, zs_val, diff_vec, plus_minus_val, word2vec, n_top, 'val')
    test_results = evals.evaluation1_acc(xs_test, ts_test, zs_test, diff_vec, plus_minus_test, word2vec, n_top, 'test')
    
    
    # Evaluation (Stability)
    print('\ncalculating stability score...')
    z_id = zs[0] # Only Gender
    assert len(zs[zs==zs[0]])==len(zs), 'evaluation1_stb は複数種類の属性には未対応 '
    if plus_minus_or_info=='use-knowledge':  
        stb_results = 'N/A'
        print('Can not caluculate stability')
    elif plus_minus_or_info in ['minus-all', 'plus-all']:
        stb_results = evals.evaluation1_stb(diff_vec, plus_minus_stb, z_id, word2vec, M, F, stb_sampling_test, n_top)
    else:
        assert False
    
    # Save results
    print('\nsaving results and experimental settings...')
    del settings['word2vec']
    results = {}
    results['experimental_settings'] = settings
    #results['diff_vec'] = diff_vec
    results['plus_minus_val']  = plus_minus_val
    results['plus_minus_test'] = plus_minus_test
    results['accuracy'] = {'train':'No train acc because this is not lean-based method', 
                           'val':val_results, 
                           'test':test_results}
    results['stability'] = {'score':stb_results, 
                            'stb_sampling_test':stb_sampling_test}
    path_save = '{}/result/{}/evaluation_results'.format(out, experiment_name)
    os.makedirs(path_save, exist_ok=True)
    with open('{}/evaluation_results.json'.format(path_save), 'w') as f:
        json.dump(results, f)

    print('done.')
    
    
    
    
def experiment1_one_diff(settings):
    experiment_name = settings['experiment_name']
    word2vec = settings['word2vec']
    stb_sampling_train = 0 
    n_argument = 0 
    plus_minus_or_info = settings['model']['plus_minus_or_info'] 
    settings['model']['name'] = 'one_diff'
    datasets = settings['datasets']
    path_dataset = settings['path_dataset']
    stb_sampling_test = settings['stb_sampling_test'] 
    n_top = settings['n_top'] 
    out = settings['out']
    emb_model = settings['embedding_method']
    
    # Load data
    xs_train, xs_val, xs_test, ts_train, ts_val, ts_test, zs_train, zs_val, zs_test, M, F, xs, ts, zs, A_train, A_val, A_test, z_ids, N_train = data.load(datasets, word2vec, stb_sampling_train, path_dataset, emb_model)
            
    # Get diff vector
    ms = xs_train[::2]
    ws = ts_train[::2]
    diff_vecs, pairs = get_one_diff_vecs(word2vec, ms, ws)

    # Get plus-minus list
    plus_minus_val, plus_minus_test, plus_minus_stb = get_plus_minus_list(xs_val, xs_test, stb_sampling_test, plus_minus_or_info, M, F)

    # Evaluation (Accuracy)
    result_list = []
    val_sota_pairs = {}
    val_sota = -1.0
    for diff_vec, pair in zip(diff_vecs, pairs):
        one_result = {}
        pair_diff = '{}-{}'.format(pair[0], pair[1])
        print('\ncalculating val accuracies ({}) ...'.format(pair_diff))
        val_results = evals.evaluation1_acc(xs_val, ts_val, zs_val, diff_vec, plus_minus_val, word2vec, n_top, 'val')
        one_result['diff_vec'] = pair_diff 
        one_result['accuracy'] = {'train':'No train acc. This is not a leaning-based method.', 
                                   'val':val_results, 
                                   'test':'No test acc. This diff_vec is not SOTA of val'}
        result_list.append(one_result)
        
        if val_results['mean_acc@1']>val_sota:
            val_sota_pairs = {}
            val_sota_pairs[pair_diff] = {'mean_acc':val_results,
                                         'diff_vec':diff_vec}
            val_sota = val_results['mean_acc@1']
        elif val_results['mean_acc@1']==val_sota:
            val_sota_pairs[pair_diff] = {'mean_acc':val_results,
                                         'diff_vec':diff_vec}
            val_sota = val_results['mean_acc@1']
        
    # Test
    test_sota_pairs = {}
    test_sota = -1.0
    for pair_diff, sota in val_sota_pairs.items():
        print('\ncalculating test accuracies ({}) ...'.format(pair_diff))
        diff_vec = sota['diff_vec']
        test_results = evals.evaluation1_acc(xs_test, ts_test, zs_test, diff_vec, plus_minus_test, word2vec, n_top, 'test')
        if test_results['mean_acc@1']>test_sota:
            test_sota_pairs = {}
            test_sota_pairs[pair_diff] = {'mean_acc':test_results}
            test_sota = test_results['mean_acc@1']
        elif val_results['mean_acc@1']==val_sota:
            test_sota_pairs[pair_diff] = {'mean_acc':test_results}
            test_sota = test_results['mean_acc@1']
            
    # Evaluation (Stability)
    if plus_minus_or_info=='use-knowledge':
        stb_results = 'N/A'
        best_score = -1.0
        for pair_diff in set(test_sota_pairs.keys()):
            method_score = test_sota_pairs[pair_diff]['mean_acc']['mean_acc@1'] + test_sota_pairs[pair_diff]['mean_acc']['mean_acc@2'] + test_sota_pairs[pair_diff]['mean_acc']['mean_acc@3']
            if method_score>=best_score:
                best_pair = copy.deepcopy(pair_diff)
                best_score = method_score
        best = {'pair':best_pair, 
                'accuracy':{'train':'No train acc because this is not lean-based method', 
                            'val':val_sota_pairs[best_pair]['mean_acc'], 
                            'test':test_sota_pairs[best_pair]['mean_acc']},
                'stability':{'score':'N/A', 
                             'stb_sampling_test':stb_sampling_test}}
        
    elif plus_minus_or_info in ['minus-all', 'plus-all']:   
        stb_sota_pairs = {}
        stb_sota = -1.0
        for pair_diff in set(test_sota_pairs.keys()):
            print('\ncalculating stability score...')

            z_id = zs[0] # Only Gender
            assert len(zs[zs==zs[0]])==len(zs), 'evaluation1_stb は複数種類の属性には未対応 '
        
            stb_results = evals.evaluation1_stb(diff_vec, plus_minus_stb, z_id, word2vec, M, F, stb_sampling_test, n_top)
            if stb_results['mean_stb@1']>=stb_sota:
                stb_sota_pairs[pair_diff] = {'mean_stb':stb_results}
                stb_sota = stb_results['mean_stb@1']
            
        best_score = -1.0
        for pair_diff in set(stb_sota_pairs.keys()):
            method_score = stb_sota_pairs[pair_diff]['mean_stb']['mean_stb@1'] * test_sota_pairs[pair_diff]['mean_acc']['mean_acc@1']
            if method_score>=best_score:
                best_pair = copy.deepcopy(pair_diff)
                best_score = method_score
                
        best = {'pair':best_pair, 
                'accuracy':{'train':'No train acc because this is not lean-based method', 
                            'val':val_sota_pairs[best_pair]['mean_acc'], 
                            'test':test_sota_pairs[best_pair]['mean_acc']},
                'stability':{'score':stb_sota_pairs[best_pair]['mean_stb'], 
                             'stb_sampling_test':stb_sampling_test}}
    else:
        assert False
             
    print('\nbest pair: {}'.format(best['pair']))
    for n in n_top:
        print('val@{}: {}'.format(n, best['accuracy']['val']['mean_acc@{}'.format(n)]))
    for n in n_top:
        print('test@{}: {}'.format(n, best['accuracy']['test']['mean_acc@{}'.format(n)]))  
    if not plus_minus_or_info=='use-knowledge':
        for n in n_top:
            print('stb@{}: {}'.format(n, best['stability']['score']['mean_stb@{}'.format(n)]))

    # Save results
    print('\nsaving results and experimental settings...')
    del settings['word2vec']
    results = {}
    results['experimental_settings'] = settings
    results['best_result'] = best
    results['result_list'] = result_list
    results['plus_minus_val']  = plus_minus_val
    results['plus_minus_test'] = plus_minus_test
    path_save = '{}/result/{}/evaluation_results'.format(out, experiment_name)
    os.makedirs(path_save, exist_ok=True)
    with open('{}/evaluation_results.json'.format(path_save), 'w') as f:
        json.dump(results, f)

    print('done.')