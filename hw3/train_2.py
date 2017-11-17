import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
# from keras.layers import ZeroPadding2D
# from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.optimizers import *
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
# from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
train_path = sys.argv[1]
# print(train_path,valid_path)

dim_x,dim_y = 48,48
label = []
train_in = []

label_valid = []
train_valid = []

with open(train_path,'r') as fp:
	tmp = fp.readline()
	for read in range(28709):
		data = fp.readline().strip('\r\n').replace(',',' ').split()
		a = int(data[0])
		tmp = np.array([0,0,0,0,0,0,0])
		tmp[a] = 1
		b = [float(x) for x in data[1:]]
		b = np.array(b).reshape(48,48,1)
		b /= 255
		label.append(tmp)
		train_in.append(b)

label = np.array(label)
train_in = np.array(train_in)
label_valid = np.array(label_valid)
train_valid = np.array(train_valid)
# print(label.shape,train_in.shape)

train_in = np.append(train_in,np.flip(train_in,axis = 2),axis = 0)
label = np.append(label,label,axis = 0)

train_valid,train_in = train_in[:5000],train_in[5000:]
label_valid,label = label[:5000],label[5000:]

# print(label.shape,train_in.shape)
input_shape = (dim_x,dim_y,1)
num_class = int(np.max(label) + 1)

mean,std = np.mean(train_in),np.std(train_in,ddof = 1)
# np.save('normal.npy',[mean,std])

# train_in = (train_in - mean) / std
# train_valid = (train_valid - mean)/std
# model = load_model('local_model.h5')
model = Sequential()

model.add( Conv2D(64,(5,5), activation = 'selu',input_shape =input_shape, kernel_initializer='glorot_normal')) #48*48 -> 44*44
# model.add(LeakyReLU(alpha=1./20))
model.add( MaxPooling2D((2,2)))#44 * 44 -> 22*22
model.add( BatchNormalization())
model.add( Dropout(0.3))


model.add( Conv2D(128,(3,3), activation = 'selu',kernel_initializer='glorot_normal'))#22*22 -> 20*20
# model.add(LeakyReLU(alpha=1./20)) 
model.add( MaxPooling2D((2,2)))#20*20 -> 10*10
model.add( BatchNormalization())
model.add( Dropout(0.35))


model.add( Conv2D(256,(3,3), activation = 'selu',kernel_initializer='glorot_normal')) # 10*10 -> 8*8
# model.add(LeakyReLU(alpha=1./20))   
model.add( MaxPooling2D((2,2)))#8*8 -> 4*4
model.add( BatchNormalization())
model.add( Dropout(0.4))

model.add( Conv2D(512,(3,3), activation = 'selu',kernel_initializer='glorot_normal'))#4*4 -> 2*2 
# model.add(LeakyReLU(alpha=1./20))
model.add( MaxPooling2D((2,2)))#2*2 -> 1*1
model.add( BatchNormalization())
model.add( Dropout(0.45))


model.add( Flatten() )


model.add( Dense(units = 512,activation = 'selu' , kernel_initializer='glorot_normal'))
model.add( Dropout(0.45) )

model.add( Dense(units = 256,activation = 'selu' , kernel_initializer='glorot_normal'))
model.add( Dropout(0.45) )

model.add( Dense(units = 128,activation = 'selu' , kernel_initializer='glorot_normal'))
model.add( Dropout(0.45) )

model.add( Dense(units = 64,activation = 'selu',kernel_initializer='glorot_normal' ))
model.add( Dropout(0.5) )

model.add( Dense(units = 7,activation = 'softmax',kernel_initializer='glorot_normal' ))

model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('./model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))
# model.fit(train_in,label,batch_size = 100,epochs = 20)

batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs: print('\nEpoch[%d] Train-loss=%f Train-accuracy=%f Validation-loss=%f Validation-accuracy=%f' %(batch,logs['loss'], logs['acc'],logs['val_loss'],logs['val_acc'])))

callbacks.append(batch_print_callback)

datagen = ImageDataGenerator(
	width_shift_range = 0.125,
	height_shift_range = 0.125,
	rotation_range = 10
)


datagen.fit(train_in)



model.fit_generator( datagen.flow(train_in, label, batch_size=128),verbose = 1,validation_data = (train_valid,label_valid), steps_per_epoch=len(train_in) / 128, epochs = 200,callbacks=callbacks )

model.save( "./model_final.h5" )

K.clear_session()
