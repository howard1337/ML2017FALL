import numpy as np
import sys
from keras.models import load_model


test_in = []


with open(sys.argv[1],'r') as fp:
	tmp = fp.readline()
	for read in range(7178):
		data = fp.readline().strip('\r\n').replace(',',' ').split()
		b = [float(x) for x in data[1:]]
		b = np.array(b).reshape(48,48,1)
		b /= 255
		test_in.append(b)

test_in = np.array(test_in)
mean,std = np.load('./normal.npy')
test_in = (test_in - mean) / std
# print(test_in.shape)
test_in_right = np.array(test_in)
test_in_left = np.array(test_in)
test_in_up = np.array(test_in)
test_in_down = np.array(test_in)


for i in range (7148):
	for j in range (48):
		for k in range (1,48):
			test_in_right[i][j][k][0] = test_in[i][j][k-1][0]
		for k in range (0,47):
			test_in_left[i][j][k][0] = test_in[i][j][k+1][0]
		test_in_right[i][j][0][0] = 0
		test_in_left[i][j][47][0] = 0

# print(test_in_left[0])
# print(test_in_right[0])


for i in range(3,len(sys.argv)):
	model = load_model(sys.argv[i])
	result = model.predict(test_in)
	result += model.predict(test_in_left)
	result += model.predict(test_in_right)
	print('result',i-2)


cnt = 0
with open(sys.argv[2], 'w') as fp2:
	fp2.write('id,label\n')
	for i in range(7178):
		max_value = 0
		max_index = 0
		for j in range(7):
			if result[i][j]>= max_value:
				max_value = result[i][j]
				max_index = j
		fp2.write('%d,%d\n' % (i,max_index))
print(cnt)			
