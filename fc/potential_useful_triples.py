import sys
import numpy as np
from scipy import sparse
import time
import itertools
import os
from collections import defaultdict
import random

class SelectHiddenTriples(object):
    def __init__(self, dict_path, train_path, hidden_triples_path, rule_name):
        self.data_path = dict_path
        self.train_path = train_path
        self.hidden_triples_path = hidden_triples_path
        self.ent_path = dict_path + '/entities.dict'
        self.rel_path = dict_path + '/relations.dict'
        # self.test_path = dict_path + '/test.txt'
        self.rule_path = dict_path + '/' + rule_name


        self.n_raw_relation = 0
        self.n_entity = 0
        self.rel2id = {}
        self.ent2id = {}
        self.id2rel = {}
        self.id2ent = {}

        self.rel2adj = {}

        self.rule_num = 0
        self.rule_list = []

        self.true_set = set()
        self.infer_set = set()
        self.all_inferred_set = set()

        # self.test_set = set()

        self.load_data()

    def load_data(self):
        # load relation and entity dictionary
        raw_rel2id = {}
        with open(self.rel_path, 'r') as f:
            for line in f:
                rel_id, rel = line.strip().split('\t')
                raw_rel2id[rel] = int(rel_id)
        self.n_raw_relation = len(raw_rel2id)

        for rel in raw_rel2id.keys():
            self.rel2id[rel] = raw_rel2id[rel]
            self.rel2id[rel + '_v'] = raw_rel2id[rel] + self.n_raw_relation
            self.id2rel[self.rel2id[rel]] = rel
            self.id2rel[self.rel2id[rel + '_v']] = rel + '_v'

        with open(self.ent_path, 'r') as f:
            for line in f:
                ent_id, ent = line.strip().split('\t')
                self.ent2id[ent] = int(ent_id)
                self.id2ent[int(ent_id)] = ent
        self.n_entity = len(self.ent2id)

        # build rel2adj dict
        rel2triple = {}
        with open(self.train_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                h_id, r_id, t_id = self.ent2id[h], self.rel2id[r], self.ent2id[t]
                # h_id= int(h)
                # r_id = int(r)
                # t_id = int(t)
                self.true_set.add( (h_id, r_id, t_id) )
                self.true_set.add((h_id, r_id + self.n_raw_relation, t_id))
                if r_id not in rel2triple.keys():
                    rel2triple[r_id] = []
                    rel2triple[r_id + self.n_raw_relation] = []
                rel2triple[r_id].append((h_id, t_id))
                rel2triple[r_id + self.n_raw_relation].append((t_id, h_id))

        for rel_id in rel2triple.keys():
            # adjMat = scipy.sparse.dok_matrix
            adjMat = sparse.dok_matrix((self.n_entity, self.n_entity))
            head_tail_list = rel2triple[rel_id]
            for head_id, tail_id in head_tail_list:
                adjMat[head_id, tail_id] = 1
            self.rel2adj[rel_id] = adjMat
            self.rel2adj[rel_id + self.n_raw_relation] = adjMat.transpose()  # add reverse


        # with open(self.test_path, 'r') as f:
        #     for line in f:
        #         h, r, t = line.strip().split('\t')
        #         self.test_set.add((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        # load rule
        # rule: "weight\thead_rel\tbody1_rel\tbody2_rel"
        cnt = 0
        with open(self.rule_path, 'r') as f:
            for line in f:
                splits = line.strip().split('\t')
                rule_weight = float(splits[0])
                cnt += 1
                if splits[1] not in self.rel2id.keys() or splits[2] not in self.rel2id.keys():
                    print('Rule not appliable: '+str(cnt) + line.strip())
                    continue

                head_id = self.rel2id[splits[1]]
                if len(splits) == 3:
                    body1_id = self.rel2id[splits[2]]
                    self.rule_list.append([head_id, body1_id, rule_weight])
                elif len(splits) == 4:
                    if splits[3] not in self.rel2id.keys():
                        print(str(cnt) + line.strip())
                        continue
                    body1_id = self.rel2id[splits[2]]
                    body2_id = self.rel2id[splits[3]]
                    self.rule_list.append([head_id, body1_id, body2_id, rule_weight])
        self.rule_num = len(self.rule_list)

    def random_sample(self, list, k):
        if len(list)>k:
            random.shuffle(list)
            list = list[:k]
        return list

    def single_rule (self, head_id, body1_id, body2_id,threshold):
        head_adj = self.rel2adj[head_id].tocsr()
        nonzeroRow = np.array(np.sum(head_adj,axis = 1).nonzero())[0]
        # print('nonzeroRow ', len(nonzeroRow))

        nonzeroCol = np.array(np.sum(head_adj,axis = 0).nonzero())[1]
        # print('nonzeroCol ', len(nonzeroCol))

        # h<-b1^?
        selected_body_adj1 = self.rel2adj[body1_id][nonzeroRow,:].tocsr()
        selected_body_idx1 = np.array(selected_body_adj1.nonzero()).transpose()
        candidate_head = selected_body_idx1[:,1]
        candidate_head = list(set(candidate_head))
        # print ('candidate_head', len(candidate_head))
        candidate_head = self.random_sample(candidate_head, threshold)
        selected_nonzeroCol = self.random_sample(nonzeroCol, threshold)
        candidate_idx1 = np.array(np.meshgrid(candidate_head,selected_nonzeroCol)).T.reshape(-1,2)
        candidate_idx1 = set([(row[0],row[1]) for row in candidate_idx1])

        # exlude observed triples
        body_adj2 = self.rel2adj[body2_id].tocsr()
        body_idx2 = np.array(body_adj2.nonzero()).transpose()
        body_idx2 = set([(row[0],row[1]) for row in body_idx2 ])
        candidate_idx1 = candidate_idx1 - body_idx2
        if body2_id not in self.infer_set:
            self.infer_set[body2_id] = candidate_idx1
        else:
            self.infer_set[body2_id] = self.infer_set[body2_id].union(candidate_idx1)

        # h<-?^b2
        selected_body_adj2 = self.rel2adj[body2_id][:,nonzeroCol].tocsr()
        selected_body_idx2 = np.array(selected_body_adj2.nonzero()).transpose()
        candidate_tail = selected_body_idx2[:, 0]
        candidate_tail = list(set(candidate_tail))
        # print('candidate_tail', len(candidate_tail))
        candidate_tail = self.random_sample(candidate_tail, threshold)
        selected_nonzeroRow = self.random_sample(nonzeroRow, threshold)
        candidate_idx2 = np.array(np.meshgrid(selected_nonzeroRow,candidate_tail)).T.reshape(-1,2)
        candidate_idx2 = set([(row[0],row[1]) for row in candidate_idx2])
        # print ('candidate_idx2',len(candidate_idx2))

        # exlude observed triples
        body_adj1 = self.rel2adj[body1_id].tocsr()
        body_idx1 = np.array(body_adj1.nonzero()).transpose()
        body_idx1 = set([(row[0], row[1]) for row in body_idx1])
        candidate_idx2 = candidate_idx2 - body_idx1

        if body1_id not in self.infer_set:
            self.infer_set[body1_id] = candidate_idx2
        else:
            self.infer_set[body1_id] = self.infer_set[body1_id].union(candidate_idx2)

    def eval(self,use_cuda, cuda, model_path, top_k_threshold=0.1):
        if use_cuda:
            cmd = 'CUDA_VISIBLE_DEVICES={} python ./kge/run.py --cuda ' \
                  '--do_score_calculation --data_path {} --hidden_triples_path {} ' \
                  '-init {} --top_k_percent {}'.format(cuda,
                                                        self.data_path, self.hidden_triples_path, 
                                                        model_path, top_k_threshold)
        else:
            cmd = 'python ./kge/run.py --do_score_calculation ' \
                  '--data_path {} --hidden_triples_path {} -init {} ' \
                  '--top_k_percent {}}'.format(self.data_path, self.hidden_triples_path, 
                    model_path, top_k_threshold)

        # top_k_threshold is the input parameter --top_k_percent in ../kge/run.py
        os.system(cmd)

    def run(self,threshold):
        self.infer_set = defaultdict(set)
        for rule in (self.rule_list):
            if len(rule) == 4:
                h, b1, b2 = rule[:3]
                self.single_rule(h, b1, b2, threshold)
                # print (rule)
        # print (self.infer_set)
        # print('This hop inferred %d triples.' %(len(self.infer_set)))
        cnt= 0
        with open(self.hidden_triples_path+'/hidden.txt', 'w') as w:
            for r in self.infer_set:
                if r< self.n_raw_relation:
                    for (h,t) in self.infer_set[r]:
                        hh = self.id2ent[h]
                        rr = self.id2rel[r]
                        tt = self.id2ent[t]
                        w.write(str(hh) + '\t' +str(rr) + '\t' + str(tt) + '\n')
                        cnt+=1
                else:
                    for (h,t) in self.infer_set[r]:
                        hh = self.id2ent[h]
                        rr = self.id2rel[r-self.n_raw_relation]
                        tt = self.id2ent[t]
                        w.write(str(tt) + '\t' +str(rr) + '\t' + str(hh) + '\n')
                        cnt+=1
        print('This hop inferred %d triples.' %(cnt))


# if __name__ == '__main__':
    # # start_time = time.time()

    # # dataset = sys.argv[1]
    # data_path = '../data/kinship' # dataset
    # train_path = '../data/kinship/train.txt' 
    # hidden_triples_path = '../models/kinship/' # path to store selected triple
    # model_path = '../models/kinship/' # model path
    # rule_name = 'MLN_rule.txt'
    # use_cuda = True
    # cuda = 2
    # model = SelectHiddenTriples(data_path, train_path, hidden_triples_path, rule_name)

    # model.run(threshold=50) # find all hidden triple # threshold (50)
    # # end_time = time.time()

    # # print('Time used: ' + str(end_time - start_time) + ' seconds.')

    # start_time = time.time()
    # model.eval(use_cuda, cuda, model_path, top_k_threshold=0.1)
    # end_time = time.time()

    # print('Time used: ' + str(end_time - start_time) + ' seconds.')
