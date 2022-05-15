import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import *

labels=[]
with open('fc_test.txt') as f:
	for line in f:
		h,r,t,label=line.strip().split()
		labels.append(int(label))
# labels=np.array(labels)

scores=[]
with open('pred_score.txt') as f:
	for line in f:
		h,r,t,score=line.strip().split()
		scores.append(float(score))
# scores=np.array(scores)



df=pd.DataFrame({'score':scores, 'label':labels})
# print(df)

pos_scores=df[df['label']==1]['score']
neg_scores=df[df['label']==0]['score']

# plt.hist(scores)

# plt.show()

# plt.hist(neg_scores)
# plt.show()
thres=-10

max_f1=0
p,r = 0,0
th=-10
while thres<0:
	# print('Threshold: ', thres)
	y_pred=[]
	for s in scores:
		if s>=thres:
			y_pred.append(1) 
		else:
			y_pred.append(0)
	precision = precision_score(labels, y_pred)
	recall = recall_score(labels, y_pred)
	f1 = f1_score(labels, y_pred)

	if f1>max_f1:
		max_f1=f1
		p=precision
		r=recall
		th=thres

	# # print('Hop: ', k)
	# print('Precision: ', precision)
	# print('Recall: ', recall)
	# print('F1: ', f1)
	thres+=0.01

print('For max_f1...')
print('Precision: ', p)
print('Recall: ', r)
print('F1: ', max_f1)
print('Threshold: ', th)

