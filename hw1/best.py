import numpy
import math
import time
import random
from sklearn.utils import shuffle
import sys


train_csv = sys.argv[1]
test_csv = sys.argv[2]
ans_csv = sys.argv[3]

month = 12
days = 20

train_in = []
train_out = []

# calculate error
def cal(ans, out):
	return numpy.sqrt(numpy.sum((numpy.abs(ans - out)) * (numpy.abs(ans - out)))) / numpy.sqrt(len(ans))

# get features
def get(data):
	new = []
	for i in range(19,28): #CO
		new.append(data[i])
	for i in range(19,28): #CO
		new.append(data[i]**2)
	for i in range(19,28): #CO
		new.append(data[i]**3)

	for i in range(64,73): #O3
		new.append(data[i])
	for i in range(64,73): #O3
		new.append(data[i]**2)
	for i in range(64,73): #O3
		new.append(data[i]**3)
	for i in range(73,91): #PM10 + PM2.5
		new.append(data[i])
	for i in range(73,91): #PM10 + PM2.5
		new.append(data[i]**3)
	for i in range(91,100):
		new.append(data[i])
	for i in range(109,118):#SO2
		new.append(data[i])
	for i in range(109,118):#SO2
		new.append(data[i]**2)

	for i in range(127,163):
		new.append(data[i])
	for i in range(145,163):
		new.append(data[i])
	new.append(data[0])	
	return new	

# read data ( total 5652 training data )
with open(train_csv,'r',encoding = 'big5') as fp:
	tmp = fp.readline()
	for m in range(month): 
		inin = [[] for i in range(18)]
		
		for d in range(days):
			for i in range(18):
				array = fp.readline().replace('NR','0').replace('\n','').split(',')
				del array[0:3]
				if m == 6 :
					continue
				inin[i] += (array)
	
		cnt = len(inin[0])
		for i in range(9,cnt):
			data = [1]
			#data [type of feature] [hour]
			for j in range(18):
				#18 types of feature
				for k in range(9):
					# first nine hours
					data.append(float(inin[j][i-9+k]))
			train_in.append(get(data))
			# (i) the tenth hour 
			train_out.append([float(inin[9][i])])

train_in = numpy.array(train_in)
train_out = numpy.array(train_out)

FFF = len(train_in[0])
print(FFF)
# train 
mod = numpy.array([[0.0] for i in range(FFF)])
rate = numpy.array([[70] for i in range(FFF)])
grad_t = numpy.array([[0.0] for i in range(FFF)])

div = []
std = []
for i in range(FFF):
	div.append(numpy.average(train_in[:,[i]]))
	std.append(numpy.std(train_in[:,[i]], ddof=1))
	if std[i] == 0:
		std[i] = 1
	print (std[i])

for i in range(FFF):
	train_in[:,[i]] = (train_in[:,[i]] - div[i] ) / std[i] + 0.05

#print (len(train_in.dot(mod)))
#print (cal(train_in.dot(mod), train_out))

best = 10000
now_time = time.time()
for T in range(1600):

	predict = train_in.dot(mod)
	#print(predict,numpy.sqrt(predict))
	error =  cal((predict),(train_out))
	if abs(error - best) < 1e-6:
		print ('T = {}'.format(T))
		break
	if error < best:
		best = error
		print ('error = ',error)

	grad = numpy.transpose(train_in).dot(predict - train_out) 
	grad_t += grad * grad
	mod -= rate * grad / (grad_t ** 0.5)

with open(test_csv, 'r') as fp, open(ans_csv, 'w') as fp2:
	flag = 1
	cnt = 0
	fp2.write('id,value\n')
	while True:
		test = [1]
		for i in range(18):
			instr = fp.readline()
			if len(instr) == 0:
				flag = 0
				break
			arr = instr.replace('NR','0').replace('\n','').split(',')
			del arr[0:2]
			test += arr
		if flag == 0:
			break
		for i in range(len(test)):
			test[i] = float(test[i])
		test = get(test)
		for i in range(FFF):
			test[i] = (test[i] - div[i]) / std[i] + 0.05
		test = numpy.array([test])
		output = 'id_%d,%f\n' % (cnt, test.dot(mod)[0])
		fp2.write(output)
		cnt += 1
