import sys

def test_without_infer(data_path='../data/kinship/',model_path='../record/kinship_test_1/0/', new_test_path='./'):
    test_set=set()
    with open(data_path+'test.txt', 'r') as f:
        for line in f:
            h,r,t=line.strip().split()
            test_set.add((h,r,t))

    infer_set = set()
    with open(model_path + 'infer.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split()
            infer_set.add((h, r, t))

    new_test_set = test_set-infer_set

    with open(new_test_path+'kge_test.txt', 'w' ) as w:
        for h,r,t in new_test_set:
            w.write(h+'\t' + r+'\t'+t+'\n')


if __name__ =='__main__':
    test_without_infer()

