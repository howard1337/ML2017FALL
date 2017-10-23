import numpy
import math
import time
import random
import sys

train_X = sys.argv[1]
train_Y = sys.argv[2]
test_csv = sys.argv[3]
ans_csv = sys.argv[4]

test_out = []
train_out = []
train = []
vi = []

constant = 0.02
def acc(ans, out):
    cnt = 0
    size = len(ans)
    for i in range(size):
        if out[i] == numpy.round(ans[i] - constant):
            cnt += 1
    return cnt / size

def sig(z):
	return 1 / (1 + numpy.exp(-z))

def get(data):
	data.append(data[0]**1.5) # age
	data.append(data[0]**2) #age
	for j in range(1,7): #gain
		data.append(data[3]**(0.5*j))
	
	for j in range(1,8): #loss
		data.append(data[4]**(0.5*j))
	data.append(data[5]**2) #work hour
	if data[14] == 1 :	data[9] = 1
	if data[52] == 1 :	data[47] = 1
	if data[105] == 1 :	data[102] = 1
	for i in [0,3,4,5]:
		data.append(math.log(1 + data[i]) ** 2)
	data.append(1)

	for i in [105,84,80,78,75,68,66,22]:
		del data[i]
	return data	

with open(train_X,'r',encoding = 'big5') as fp1, open(train_Y,'r') as fp2:
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

with open(test_csv, 'r') as fp1, open('adult.csv','r') as fp2:
    fp1.readline()
    for i in range(16281):
        b = fp2.readline().replace('\n','').split(',')
        a = fp1.readline().replace('\n','').split(',')
        a = get([float(x) for x in a])
        b = [int(x) for x in b]

        vi.append(a)
        test_out.append(b)
vi = numpy.array(vi)
train = numpy.array(train)
train_out = numpy.array(train_out)

print(train.shape,train_out.shape)

FFF = len(train[0])
print(FFF)
mod = numpy.array([[0.0]for j in range(FFF)])
rate = numpy.array([[0.01]for j in range(FFF)])
grad_t = numpy.array([[0.0]for j in range(FFF)])

m_t = numpy.array([[0.0]for j in range(FFF)])
v_t = numpy.array([[0.0]for j in range(FFF)])
m_t_hat = numpy.array([[0.0]for j in range(FFF)])
v_t_hat = numpy.array([[0.0]for j in range(FFF)])

beta_m = 0.99
beta_v = 0.999
em = 1e-8
div = []
std = []

for i in range(FFF):
	div.append(numpy.average(train[:,[i]]))
	std.append(numpy.std(train[:,[i]],ddof = 0))
	if std[i] == 0:
		std[i] = 1
special = [0.0 for i in range(16281)]
special_2 =[0.0 for i in range(16281)]
for i in range(16281):
	if vi[i][89] == 1:
		special[i] = 1
	if vi[i][33] == 1:
		special_2[i] = 1




for i in range(FFF -1):
	train[:,[i]] = (train[:,[i]] - div[i]) / std[i]
	vi[:,[i]] = (vi[:,[i]] - div[i]) / std[i]

best = 0
nice = 0
idd = 0
lamda = 0

out = sig(vi.dot(mod))
acc_max = 0
accurate = 0
for T in range(2536):
	Fwb = sig(train.dot(mod))
	grad = numpy.transpose(train).dot(Fwb - train_out) + lamda * mod 
	m_t = beta_m * m_t + (1 - beta_m) * grad
	v_t = beta_v * v_t + (1 - beta_v) * (grad **2)
	m_t_hat = m_t / (1 - beta_m ** (T+1))
	v_t_hat = v_t / (1 - beta_v ** (T+1))
	#grad_t += grad * grad
	mod -= rate * m_t_hat / ((v_t_hat**0.5) + em)
	# if T >= 2450 and T % 5 == 0:
	# 	out = sig(vi.dot(mod))
	# 	error = acc(Fwb ,train_out) # ans out
	# 	print('epoch = ',T,'training error = ',error)
	# 	error = acc(out ,test_out) # ans out
	# 	print('epoch = ',T,'test error = ',error)
	# 	print()
	# 	if error > acc_max:
	# 		acc_max = error
	# 		accurate = T
print(accurate,acc_max)

out = sig(vi.dot(mod))
error = acc(Fwb ,train_out) # ans out
print('training error = ',error)

error = acc(out ,test_out) # ans out
print('test error = ',error)

with open(ans_csv, 'w') as fp2:
	fp2.write('id,label\n')
	for i in range(16281):
		if out[i] >= 0.5 + constant :
			fp2.write('%d,%d\n' % (i+1,1))
		else:
			fp2.write('%d,%d\n' % (i+1,0))