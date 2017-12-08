from gensim.utils import tokenize
from gensim.models import Word2Vec

import keras.backend as K
from keras.layers.recurrent import GRU
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, ConvLSTM2D, Reshape, MaxPooling2D, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

import sys
import numpy as np
from sklearn.utils import shuffle
import pickle

train,train_valid = [],[]
label,label_valid = [],[]

max_length = 55

with open('embeddings_matrix.txt','rb') as fp1 , open('word2idx.txt','rb') as fp2:
	embeddings_matrix = pickle.load(fp1)
	word2idx = pickle.load(fp2)
	

EMBEDDING_DIM = 128 #词向量维度

train,train_valid = [],[]
label,label_valid = [],[]
with open(sys.argv[1],'r',encoding = 'utf8') as fp:
	for line in fp:
		line = line.replace('?•','').encode('ascii',errors = 'ignore')
		line = str(line)
		a =  line.strip('\r\n').replace('+++$+++','').replace('b\"','').replace('b\'','').split()	
		b = np.zeros(max_length)
		for index,i in enumerate(a[1:]):
			if index >= max_length:
				break
			tmp = i[0]
			for k in range(1,len(i)-1):
				if i[k] != i[k-1] or i[k] != i[k+1]:
					tmp += i[k]
			if i[-1] != '\n':
				tmp += i[-1]
			if tmp in word2idx:
				b[index] = word2idx[tmp]
		# print(b)
		train.append(b)
		label.append(int(a[0]))


train,label = shuffle(train,label)

train,train_valid = train[:160000],train[160000:]
label,label_valid = label[:160000],label[160000:]

# with open('pseudo_label.txt','r',encoding = 'utf8') as fp:
# 	for line in fp:
# 		line = line.replace('?•','').encode('ascii',errors = 'ignore')
# 		line = str(line)
# 		a =  line.strip('\r\n').replace('+++$+++','').replace('b\"','').replace('b\'','').split()	
# 		b = np.zeros(max_length)
# 		for index,i in enumerate(a[1:]):
# 			if index >= max_length:
# 				break
# 			tmp = i[0]
# 			for k in range(1,len(i)-1):
# 				if i[k] != i[k-1] or i[k] != i[k+1]:
# 					tmp += i[k]
# 			if i[-1] != '\n':
# 				tmp += i[-1]
# 			if tmp in word2idx:
# 				b[index] = word2idx[tmp]
# 		# print(b)
# 		train.append(b)
# 		label.append(int(a[0]))



train = np.array(train)
label = np.array(label)
train_valid = np.array(train_valid)
label_valid = np.array(label_valid)

print(train.shape)

embedding_layer = Embedding(len(embeddings_matrix),EMBEDDING_DIM,input_length = max_length,weights=[embeddings_matrix],trainable=True)

model_ans = Sequential()
model_ans.add(embedding_layer)
model_ans.add(Bidirectional(GRU(128,activation = 'tanh',kernel_initializer = 'Orthogonal',return_sequences = True)))
model_ans.add(Dropout(0.4))
model_ans.add(Bidirectional(GRU(128,activation = 'tanh',kernel_initializer = 'Orthogonal',return_sequences = False)))
model_ans.add(Dropout(0.4))


model_ans.add(Dense(128,activation = 'relu'))
model_ans.add(Dropout(0.5))
model_ans.add(Dense(1,activation = 'sigmoid')) #16551
model_ans.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
model_ans.summary()

callbacks = []
callbacks.append(ModelCheckpoint('model-{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only= False, mode = 'auto', period=1))
model_ans.fit(train,label,verbose = True,epochs = 5,batch_size = 512,validation_data=(train_valid,label_valid),callbacks = callbacks)
