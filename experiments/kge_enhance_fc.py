import sys
import random
from sklearn.metrics import *

def negative_sampling(datapath='../data/kinship/'):

    true_triples = set()
    ent = set()
    with open(datapath + 'kg.txt', 'r') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            true_triples.add((h, r, t))
            ent.add(h)
            ent.add(t)
    with open(datapath + 'train.txt', 'r') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            true_triples.add((h, r, t))
    with open('all_infer.txt', 'r') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            true_triples.add((h, r, t))

    ent = list(ent)
    random.shuffle(ent)
    num = len(ent)
    with open('fc_test.txt', 'w') as w:
        with open(datapath + 'test.txt', 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                hflag = random.randint(0, 1)
                ind = random.randint(0, num - 1)
                if hflag == 1:
                    while ((ent[ind], r, t) in true_triples):
                        ind = random.randint(0, num - 1)
                    hh = ent[ind]
                    w.write(h + '\t' + r + '\t' + t + '\t1\n')
                    w.write(hh + '\t' + r + '\t' + t + '\t0\n')
                elif hflag == 0:
                    while ((h, r, ent[ind]) in true_triples):
                        ind = random.randint(0, num - 1)
                    tt = ent[ind]
                    w.write(h + '\t' + r + '\t' + t + '\t1\n')
                    w.write(h + '\t' + r + '\t' + tt + '\t0\n')

def eval_precision_and_recall(pred_path_list=['../record/kinship_test/0/infer.txt'], k=0):
    test_list=[]
    y_true = []
    with open('fc_test.txt', 'r') as f:
        for line in f:
            h,r,t,label=line.strip().split()
            test_list.append((h,r,t))
            y_true.append(int(label))

    pred_set = set()
    for pred_path in pred_path_list:
        with open(pred_path, 'r') as f:
            for line in f:
                h,r,t=line.strip().split()
                pred_set.add((h,r,t))

    y_pred = []
    for h,r,t in test_list:
        if (h,r,t) in pred_set:
            y_pred.append(1)
        else:
            y_pred.append(0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print('Hop: ', k)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)



if __name__ =='__main__':
    negative_sampling()
    eval_precision_and_recall(['all_infer.txt'], 0)

