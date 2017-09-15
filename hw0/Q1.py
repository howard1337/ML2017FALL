import sys
file = open(sys.argv[1],'r')
file_2 = open('Q1.txt', 'w')
inin = file.read().split()

data = []
index = []
count = []

for e,i in enumerate(inin):
	if not (i in data):
		data.append(i)
		count.append(1)
	else:
		count[data.index(i)] += 1


for i in range(len(data)):
	file_2.write(("%s %d %d") % (data[i],i,(count[i])))
	if i != len(data) - 1:
		file_2.write("\n")