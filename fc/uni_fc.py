import sys
import numpy as np
from scipy import sparse
import time


class ForwardChain(object):
    def __init__(self, dict_path, train_path, save_path, rule_name):
        self.train_path = train_path
        self.ent_path = dict_path + '/entities.dict'
        self.rel_path = dict_path + '/relations.dict'
        self.test_path = dict_path + '/test.txt'
        self.rule_path = dict_path + '/' + rule_name

        self.save_path = save_path

        self.n_raw_relation = 0
        self.n_entity = 0
        self.rel2id = {}
        self.ent2id = {}
        self.id2rel = {}
        self.id2ent = {}

        self.rel2adj = {}

        self.rule_num = 0
        self.rule_list = []

        self.train_set = set()
        self.infer_set = set()
        self.all_inferred_set = set()

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
                # if self.dataset == 'wn18rr':
                #     h = str(int(h))
                #     t = str(int(t))
                h_id, r_id, t_id = self.ent2id[h], self.rel2id[r], self.ent2id[t]
                self.train_set.add( (h_id, r_id, t_id) )
                self.train_set.add((h_id, r_id + self.n_raw_relation, t_id))
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

    def get_2N(self, head_id, tail_id):
        # head_adj = self.rel2adj[head_id]
        body_adj = self.rel2adj[tail_id]

        # head_idx = np.array(head_adj.nonzero()).transpose()
        body_idx = np.array(body_adj.nonzero()).transpose()

        for idx in body_idx:
            x, y = idx
            if head_id > self.n_raw_relation:
                self.infer_set.add((y, head_id - self.n_raw_relation, x))
            else:
                self.infer_set.add((x, head_id, y))


    def get_3N(self, head_id, body1_id, body2_id):
        # head_adj = self.rel2adj[head_id]
        body1_adj = self.rel2adj[body1_id].tocsr()
        body2_adj = self.rel2adj[body2_id].tocsr()

        body_adj = body1_adj * body2_adj

        # head_idx = np.array(head_adj.nonzero()).transpose()
        body_idx = np.array(body_adj.nonzero()).transpose()

        for idx in body_idx:
            x, y = idx
            if head_id > self.n_raw_relation:
                self.infer_set.add((y, head_id-self.n_raw_relation, x))
            else:
                self.infer_set.add((x, head_id, y))

    def one_hop(self):
        self.infer_set = set()
        for rule in (self.rule_list):
            if len(rule) == 3:
                h, b1 = rule[:2]
                self.get_2N(h, b1)

            elif len(rule) == 4:
                h, b1, b2 = rule[:3]
                self.get_3N(h, b1, b2)
        with open(self.save_path, 'w') as w:
            for h, r, t in self.infer_set:
                hh = self.id2ent[h]
                rr = self.id2rel[r]
                tt = self.id2ent[t]
                w.write(str(hh) + '\t' + str(rr) + '\t' + str(tt) + '\n')
        print('%d triples inferred in one hop.' % len(self.infer_set))

    def eval(self):
        true_triples = set()
        with open(self.test_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                true_triples.add((h,r,t))
        infer_triples = set()
        with open(self.save_path, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                infer_triples.add((h, r, t))
        acc = float(len(true_triples & infer_triples)) / float(len(true_triples))
        print('acc: %f' % acc)


    def run(self):
        cnt = 0

        while True:
            self.infer_set = set()
            for rule in (self.rule_list):
                if len(rule) == 3:
                    h, b1 = rule[:2]
                    self.get_2N(h, b1)

                elif len(rule) == 4:
                    h, b1, b2 = rule[:3]
                    self.get_3N(h, b1, b2)
            cnt +=1
            tmp_set = self.infer_set - self.all_inferred_set
            if len(tmp_set) == 0:
                break
            else:
                self.all_inferred_set |= tmp_set
                for triple in tmp_set:
                    h, r, t = triple
                    rev_r = (r + self.n_raw_relation) % (self.n_raw_relation * 2)

                    self.rel2adj[r][h, t] = 1
                    self.rel2adj[rev_r][t, h] = 1
        print('%d triples inferred in the full process.' % len(self.all_inferred_set))


    def save_infer_train(self):
        self.all_inferred_set |= self.train_set
        with open(self.save_path, 'w') as w:
            for h, r, t in self.all_inferred_set:
                hh = self.id2ent[h]
                rr = self.id2rel[r]
                tt = self.id2ent[t]
                w.write(str(hh) + '\t' + str(rr) + '\t' + str(tt) + '\n')

if __name__ == '__main__':
    dataset = sys.argv[1]
    if (len(sys.argv) == 3):
        rule_name = sys.argv[2]
    else:
        rule_name = 'MLN_rule.txt'
    ourmln = ForwardChain('./data/' + dataset + '/','./data/' + dataset + '/train.txt', rule_name)

    start_time = time.time()
    ourmln.run()
    end_time = time.time()
    print('Time used: ' + str(end_time - start_time) + ' seconds.')

    ourmln.save_infer_train()