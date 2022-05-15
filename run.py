import sys
import os
import numpy as np
import fc.fc as fc
import fc.potential_useful_triples as pt

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_kge(path, train_file_path, model='TransE', model_name='kge_model', kge_iters=10000, use_cuda=True):


    if use_cuda:

        cmd = 'CUDA_VISIBLE_DEVICES={} python kge/run.py --cuda --do_train --do_valid --do_test ' \
              '--model {} --data_path {} -b {} -n {} -d {} -g {} ' \
              '-a {} -adv -lr {} --max_steps {} --test_batch_size {} ' \
              '-save {} --train_path {}'.format(
            cuda, model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma,
            kge_alpha, kge_lr, kge_iters, kge_tbatch,
            path + '/' + model_name, train_file_path)
    else:
        cmd = 'python kge/run.py --do_train --do_valid --do_test ' \
              '--model {} --data_path {} -b {} -n {} -d {} -g {} ' \
              '-a {} -adv -lr {} --max_steps {} --test_batch_size {} ' \
              '-save {} --train_path {}'.format(
            model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma,
            kge_alpha, kge_lr, kge_iters, kge_tbatch,
            path + '/' + model_name, train_file_path)

    os.system(cmd)

def run_fc(workspace_path, train_file_name, save_name):
    FC = fc.ForwardChain(data_path, workspace_path+train_file_name, workspace_path+save_name, 'MLN_rule.txt')
    FC.run()

def eval_and_eliminate(path, k, model, model_name, train_name, save_name, noise_threshold=0.1, use_cuda=True):
    # read train.txt, write new_train.txt
    # read infer.txt, write new_infer.txt

    # print('eval_and_eliminate')
    workspace_path = path + '/' + str(k) + '/'
    

    if noise_threshold==0:
        cmd = 'cp {}/{} {}/{}'.format(workspace_path, train_name, workspace_path, save_name)
    else:
        if use_cuda:
            cmd = 'CUDA_VISIBLE_DEVICES={} python -u kge/run.py --cuda --do_eval ' \
                  '--model {} -init {} --train_path {} --noise_threshold {}  ' \
                  '--eliminate_noise_path {} --data_path {}'.format(
                cuda, model, path + '/' + model_name, workspace_path + '/' + train_name,
                noise_threshold, workspace_path + '/' + save_name, data_path)
        else:
            cmd = 'python -u kge/run.py --do_eval --model {} -init {} --train_path {} ' \
                  '--noise_threshold {}  --eliminate_noise_path {} --data_path {}'.format(
                model, path + '/' + model_name, workspace_path + '/' + train_name,
                noise_threshold, workspace_path + '/' + save_name, data_path)


    os.system(cmd)

def run_potential(workspace_path, data_path='./data/kinship',
                  train_path='./data/kinship/train.txt', rule_name='MLN_rule.txt',
                  top_k_threshold = 0.1,
                  use_cuda=True, cuda=2):
    workspace_path=workspace_path


    model = pt.SelectHiddenTriples(data_path, train_path,
                                   hidden_triples_path=workspace_path,
                                   rule_name=rule_name)

    model.run(threshold=50)  # find all hidden triple # threshold (50)

    # model.eval(use_cuda, cuda, model_path=workspace_path+'/kge_model/')
    model.eval(use_cuda, cuda, model_path=path+'/kge_model/', top_k_threshold=top_k_threshold)


if __name__ == '__main__':
    dataset = sys.argv[1]
    data_path = './data/' + sys.argv[1] + '/'

    cuda = sys.argv[2]
    if cuda == '-1':
        use_cuda = False
    else:
        use_cuda = True

    record_name = sys.argv[3]
    path = './record/' + record_name + '/'
    check_path(path)

    kge_model = sys.argv[4]
    

    iterations = int(sys.argv[5])

    noise_threshold = float(sys.argv[6])

    top_k_threshold = float(sys.argv[7])

    if len(sys.argv) > 8:
        is_init = int(sys.argv[8])
    else:
        is_init = 0

    

    kge_batch = 512
    kge_neg = 128
    kge_dim = 500
    kge_gamma = 6.0
    kge_alpha = 0.5
    kge_lr = 0.0005
    kge_iters = 5000
    final_kge_iters = 80000
    kge_tbatch = 8
    kge_reg = 0.000001

    if dataset == 'kinship':
        if kge_model == 'TransE':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 100, 24, 1, 0.001, 80000, 50000, 16, 0.0
        if kge_model == 'DistMult':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 24, 1, 0.001, 50000, 50000, 16, 0.0


    check_path(path + '/0/')
    os.system('cp {}/train.txt {}/fc_train.txt'.format(data_path, path+'/0/'))

    kge_data_path=path+'/kge_data/'
    check_path(kge_data_path)

    for k in range(iterations):

        workspace_path = path + '/' + str(k) + '/'
        check_path(workspace_path)

        print('Start Foward Chaining...')
        run_fc(workspace_path, 'fc_train.txt', 'infer.txt')
        os.system(
            'cat {}/fc_train.txt {}/infer.txt >> {}/observed.txt'.format(workspace_path, workspace_path,
                                                                             workspace_path))

        if k==0 and is_init==0:
            print('Start KGE Training...')
            os.system(
            'cat {}/fc_train.txt {}/infer.txt >> {}/kge_train.txt'.format(workspace_path, workspace_path,
                                                                             kge_data_path))
            run_kge(path, kge_data_path+'/kge_train.txt', kge_model, 'kge_model', kge_iters=kge_iters, use_cuda=use_cuda)


        print('Start Eval and Eliminating...')
        eval_and_eliminate(path, k, kge_model, 'kge_model', 'observed.txt', 'new_observed.txt',
                           noise_threshold=noise_threshold, use_cuda=use_cuda)

        print('Start Finding Potential Useful Triples...')
        run_potential(workspace_path=workspace_path, data_path=data_path,
                      train_path=workspace_path+'new_observed.txt', rule_name='MLN_rule.txt', 
                      top_k_threshold = top_k_threshold,
                      use_cuda=use_cuda, cuda=cuda)

        next_workspace_path = path + '/' + str(k+1) + '/'
        check_path(next_workspace_path)
        os.system('cat {}/new_observed.txt  {}/selected_triples.txt >> {}/fc_train.txt'.format(
        	workspace_path, workspace_path, next_workspace_path))


    workspace_path = path + '/' + str(iterations) + '/'
    run_kge(path, workspace_path+'/fc_train.txt', model=kge_model, model_name='final_model', 
    	kge_iters=kge_iters, use_cuda=use_cuda)
