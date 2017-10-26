import numpy
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score

import math
import sys

train_X = sys.argv[1]
train_Y = sys.argv[2]
test_csv = sys.argv[3]
ans_csv = sys.argv[4]
FFF = 0

train_out = []
train = []
vi = []


def train_xgboost():
	best_x,best_y,best_acc = 0,0,0
	for x in range(17,18):
		for y in range(4,5):
			model = XGBClassifier(max_depth = 3,n_estimators = 100 * x,learning_rate = 0.01 * y)
			model.fit(train,train_out.flatten())
			train_pred = model.predict(train)
			# acc_train = accuracy_score(train_out.flatten(),train_pred)
			print('x = %d, y = %d Train Accuracy: %.2f%%'%(x,y,acc_train * 100.0))
	return model


def get(data):
	data.append(data[0]**1.5) # age
	data.append(data[0]**2) #age
	for j in range(1,8): #gain
		data.append(data[3]**(0.5*j))
	for j in range(1,8): #loss
		data.append(data[4]**(0.5*j))
	data.append(data[5]**2) #work hour
	for i in [0,3,4,5]:
		data.append(math.log(1 + data[i]) ** 2)
	data.append(1)
	if data[14] == 1 :	data[9] = 1
	if data[52] == 1 :	data[47] = 1
	if data[105] == 1 :	data[102] = 1
	for i in [105,84,80,78,75,68,66,22]:
		del data[i]
	return data	

with open(train_X,'r') as fp1, open(train_Y,'r') as fp2:
	tmp = fp1.readline()
	tmp = fp2.readline()
	while True:
		a = fp1.readline().replace('\n','').split(',')
		b = fp2.readline().replace('\n','').split(',')
		if len(a) == 1:
			break
		a = get([float(x) for x in a])
		b = [int(x) for x in b]
		train.append(a)
		train_out.append(b)

with open(test_csv, 'r') as fp1:
    fp1.readline()
    for i in range(16281):
        a = fp1.readline().replace('\n','').split(',')
        a = get([float(x) for x in a])
       
        vi.append(a)
       
vi = numpy.array(vi)
train = numpy.array(train)
train_out = numpy.array(train_out)
FFF = len(train[0])

# print(FFF)
mod = numpy.array([[0.0]for j in range(FFF)])
rate = numpy.array([[1]for j in range(FFF)])
grad_t = numpy.array([[0.0]for j in range(FFF)])

div = []
std = []

for i in range(FFF):
	div.append(numpy.average(train[:,[i]]))
	std.append(numpy.std(train[:,[i]],ddof = 1))
	if std[i] == 0:
		std[i] = 1
# for i in range(FFF -1):
# 	train[:,[i]] = (train[:,[i]] - div[i]) / std[i]
# 	vi[:,[i]] = (vi[:,[i]] - div[i]) / std[i]

model = train_xgboost()
out = model.predict(vi)
# if not os.path.exists('./output'):
# 	os.makedirs('./output')
# model.save('./output/hw2.h5')
#print(train.shape,train_out.shape)
# print('id,label')
# for i in range(16281):
# 	print('%d,%d' % (i+1,numpy.round(out[i])))
      

with open(ans_csv, 'w') as fp2:
    fp2.write('id,label\n')
    for i in range(16281):
        fp2.write('%d,%d\n' % (i+1,numpy.round(out[i])))
