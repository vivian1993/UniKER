import sys
import os
import numpy as np
import fc.fc as fc


def run_fc(data_path, train_file_path, save_file_path):
    FC = fc.ForwardChain(data_path, train_file_path, save_file_path, 'MLN_rule.txt')
    FC.run()

cmd = 'cp data/kinship/train.txt experiments/fc_train.txt'
os.system(cmd)
for i in range(0, 50):
	run_fc('data/kinship/', 'experiments/fc_train.txt', 'experiments/'+str(i)+'_infer.txt')
	cmd = 'cat experiments/fc_train.txt '+'experiments/'+str(i)+'_infer.txt'+' >> experiments/fc_train.txt'
	os.system(cmd)
	