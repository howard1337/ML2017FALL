from keras.models import load_model
import pickle
import sys
import numpy as np


max_length = 55
with open('embeddings_matrix.txt','rb') as fp1 , open('word2idx.txt','rb') as fp2:
	embeddings_matrix = pickle.load(fp1)
	word2idx = pickle.load(fp2)

model_ans = load_model(sys.argv[3])
train_hi = []
with open(sys.argv[1],'r',encoding = 'utf8') as fp:
	flag = 0
	for line in fp:
		if flag == 0:
			flag += 1
			continue
		line = line.split(',',maxsplit = 1)
		line = str(line[1])
		line = line.replace('?•','').encode('ascii',errors = 'ignore')
		line = str(line)
		a =  line.strip('\r\n').replace('b\"','').replace('b\'','').split()
		b = np.zeros(max_length)
		for index,i in enumerate(a):
			tmp = i[0]
			for k in range(1,len(i)-1):
				if i[k] != i[k-1] or i[k] != i[k+1]:
					tmp += i[k]
			if i[-1] != '\n':
				tmp += i[-1]
			if tmp in word2idx:
				b[index] = word2idx[tmp]
		train_hi.append(b)
		
print('testing')

train_hi = np.array(train_hi)
ans = model_ans.predict(train_hi,batch_size = 512,verbose = 1)
# print('second')
# for i in range(3,len(sys.argv)):
# 	model_ans = load_model(sys.argv[i])
# 	ans += model_ans.predict(train_hi,batch_size = 512,verbose = 1)

# ans /= (len(sys.argv) -2)
'''
print('confidence')
for i in range(200000):
	confidence = [0.0,0.0,0.0,0.0]
	for j in range(0,4):
		if ans_1[j][i] >= 0.5:
			confidence[j] = ans_1[j][i] - 0.5
		else:
			confidence[j] = 0.5 - ans_1[j][i]
	max_index = -1
	max_value = -1
	for j in range(0,4):
		if confidence[j] >= max_value:
			max_value = confidence[j]
			max_index = j
	ans[i][0] = ans_1[max_index][i]

'''

print('go')
with open(sys.argv[2], 'w',encoding = 'utf8') as fp2:
    fp2.write('id,label\n')
    for index,i in enumerate(ans):
        if ans[index][0] >= 0.5:
            fp2.write('%d,%d\n' % (index,1))
        elif ans[index][0] < 0.5:
            fp2.write('%d,%d\n' % (index,0))
