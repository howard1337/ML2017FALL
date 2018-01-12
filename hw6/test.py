import sys
import numpy as np

cluster = np.load('cluster.npy')

cnt = 0
zero_cnt = 0
with open(sys.argv[1],'r') as fp, open(sys.argv[2],'w') as fp2:
	fp.readline()
	print('ID,Ans',file = fp2)
	for line in fp:
		num = line.strip('\r\n').split(',')[1:]
		index_a, index_b = int(num[0]), int(num[1])
		print('%d,%d'%(cnt,cluster[index_a] == cluster[index_b]),file = fp2)
		cnt += 1
		if cluster[index_a] != cluster[index_b]:
			zero_cnt += 1
print(zero_cnt)