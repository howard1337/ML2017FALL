import numpy
import math
import time 
import random
import sys

train_csv = sys.argv[1]
test_csv = sys.argv[2]
ans_csv = sys.argv[3]
N_1 = 0
N_2 = 0
train_in = [[],[]]

train_out = []
train = []
vi = []

workclass = [' Private',' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' Local-gov',' State-gov',' Without-pay',' Never-worked',' -88']
dict_workclass = {i:e for e,i in enumerate(workclass)}

marital = [' Married-civ-spouse',' Divorced',' Never-married',' Separated',' Widowed',' Married-spouse-absent',' Married-AF-spouse',' -88']
dict_marital = {i:e for e,i in enumerate(marital)}

occupation = [' Tech-support',' Craft-repair',' Other-service',' Sales',' Exec-managerial',' Prof-specialty',' Handlers-cleaners',' Machine-op-inspct',' Adm-clerical',' Farming-fishing',' Transport-moving',' Priv-house-serv',' Protective-serv',' Armed-Forces',' -88']
dict_occupation = {i:e for e,i in enumerate(occupation)}

relationship = [' Wife',' Own-child',' Husband',' Not-in-family',' Other-relative',' Unmarried',' -88']
dict_relationship = {i:e for e,i in enumerate(relationship)}

race = [' White',' Asian-Pac-Islander',' Amer-Indian-Eskimo',' Other',' Black',' -88']
dict_race = {i:e for e,i in enumerate(race)}

dict_sex = {' Female':0,' Male':1,' -88':2}

native = [' United-States',' Cambodia',' England',' Puerto-Rico',' Canada',' Germany',' Outlying-US(Guam-USVI-etc)',' India',' Japan',' Greece',' South',' China',' Cuba',' Iran',' Honduras',' Philippines',' Italy',' Poland',' Jamaica',' Vietnam',' Mexico',' Portugal',' Ireland',' France',' Dominican-Republic',' Laos',' Ecuador',' Taiwan',' Haiti',' Columbia',' Hungary',' Guatemala',' Nicaragua',' Scotland',' Thailand',' Yugoslavia',' El-Salvador',' Trinadad&Tobago',' Peru',' Hong',' Holand-Netherlands',' -88']
dict_native = {i:e for e,i in enumerate(native)}

def acc(ans, out):
    cnt = 0
    size = len(ans)
    for i in range(size):
        if out[i] == round(1 - ans[i]):
            cnt += 1
    return cnt / size

def sig(z):
	return 1 / (1 + numpy.exp(-z))

def get(data):
	new = []
	new.append(int(data[1])) #age
	new.append(int(data[1])**2) #age
	
	#new.append(int(dict_workclass[data[2]])) # workclass
	new.append(int(data[3])) #fnlwgt	
	
	new.append(int(data[5])) #education num
	new.append(int(data[5])**2) #education num
	
	new.append(int(dict_marital[data[6]])) #marital_status
	new.append(int(dict_occupation[data[7]])) #occupation
	new.append(int(dict_relationship[data[8]])) #relationship
	
	#new.append(int(dict_race[data[9]])) #race
	new.append(int(dict_sex[data[10]])) #sex
	new.append(int(data[11])) #capital gain
	new.append(int(data[11])**2) #capital gain
	
	new.append(int(data[12])) #capital loss
	new.append(int(data[12])**2) #capital loss

	new.append(int(data[13])) #hours_per_week
	new.append(int(data[13])**2) #hours_per_week
	
	#new.append(int(dict_native[data[14]])) #native_country
	#new.append(int(data[0])) #bias
	return new	

with open(train_csv,'r',encoding = 'big5') as fp:
	tmp = fp.readline()
	while True:
		data = []
		data = fp.readline().replace('?','-88').replace('\n','').split(',')
		if len(data) == 1:
			break
		data.insert(0,1)
		train.append(get(data))
		if data[-1] == ' <=50K':
			train_in[0].append(get(data))
			train_out.append(0)
		else:
			train_in[1].append(get(data))
			train_out.append(1)
			
N_1 = len(train_in[0])
N_2 = len(train_in[1])
FFF = len(train_in[0][0])
train_in[0] = numpy.array(train_in[0])
train_in[1] = numpy.array(train_in[1])
train = numpy.array(train)
train_out = numpy.array(train_out)


avg_1 = numpy.array([0.0 for i in range(FFF)])
avg_2 = numpy.array([0.0 for i in range(FFF)])

for i in range(N_1):
	avg_1 += train_in[0][i]

for i in range(N_2):
	avg_2 += train_in[1][i]

avg_1[:] /= N_1
avg_2[:] /= N_2

sigma = numpy.array([[0.0 for i in range(FFF)]for j in range(FFF)])
sigma_1 = numpy.array([[0.0 for i in range(FFF)]for j in range(FFF)])
sigma_2 = numpy.array([[0.0 for i in range(FFF)]for j in range(FFF)])

for i in range(FFF):
	for j in range(FFF):
		if (i > j):
			sigma_1[i][j] = sigma_1[j][i]
			continue
		for k in range(N_1):
			sigma_1[i][j] += (train_in[0][k][i] - avg_1[i]) * (train_in[0][k][j] - avg_1[j])
		sigma_1[i][j] /= N_1

for i in range(FFF):
	for j in range(FFF):
		if (i > j):
			sigma_2[i][j] = sigma_2[j][i]
			continue
		for k in range(N_2):
			sigma_2[i][j] += (train_in[1][k][i] - avg_2[i]) * (train_in[1][k][j] - avg_2[j])
		sigma_2[i][j] /= N_2

for i in range(FFF):
	for j in range(FFF):
		sigma[i][j] = (sigma_1[i][j] * N_1 + sigma_2[i][j] * N_2)/(N_1 + N_2)

inverse = numpy.linalg.inv(sigma)
w = (avg_1 - avg_2).T.dot(inverse) # 1 x FFF 
b = -avg_1.T.dot(inverse).dot(avg_1)/2 + avg_2.T.dot(inverse).dot(avg_2)/2 + math.log(N_1) - math.log(N_2) # 1 x FFF
#print(train.shape) # 32561 x FFF 1xFFF x FFF x1
#print(train[0].shape)
#print(w.shape)
#print(b.shape)
cnt = 0

tmp = train.dot(w) + b
Fwb = sig(tmp)
#print(Fwb,train_out)
error = acc(Fwb ,train_out) # ans out
print(error)

with open(test_csv, 'r') as fp1:
    fp1.readline().replace('?','-88').replace('\n','').split(',')
    for i in range(16281):
        a = fp1.readline().replace('?','-88').replace('\n','').split(',')
        a.insert(0,1)
        a = get(a)
        vi.append(a)
vi = numpy.array(vi)

out = sig(vi.dot(w)+b)
with open(ans_csv, 'w') as fp2:
    fp2.write('id,label\n')
    for i in range(16281):
        fp2.write('%d,%d\n' % (i+1,round(1 - out[i])))

