import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle

import keras.backend as K
from keras.models import Model,load_model
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.layers import Input, Embedding, Reshape, Dropout,add, Dot, Lambda, Layer
from keras.preprocessing.sequence import pad_sequences

test_path = sys.argv[3]
model_path = sys.argv[2]
output_path = sys.argv[1]

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def read_data(filename, user2id, movie2id):
    df = pd.read_csv(filename)

    df['UserID'] = df['UserID'].apply(lambda x: user2id[x])
    df['MovieID'] = df['MovieID'].apply(lambda x: movie2id[x])

    return df['TestDataID'], df[['UserID', 'MovieID']].values


def submit(filename, id, pred):
    df = pd.DataFrame({'TestDataID': id, 'Rating': pred}, columns=('TestDataID', 'Rating'))
    df.to_csv(filename, index=False, columns=('TestDataID', 'Rating'))


user2id = np.load('user2id.npy')[()]
movie2id = np.load('movie2id.npy')[()]
id, X_test = read_data(test_path, user2id, movie2id)
mean, std = np.load('mean.npy'), np.load('std.npy')
# pred_en = []
model = load_model(model_path, custom_objects={'rmse': rmse})

print('std = ',std)
pred = model.predict([X_test[:, 0], X_test[:, 1]],verbose = 1).squeeze()
pred = (pred * std) + mean
pred = pred.clip(1.0, 5.0)
# pred_en.append(pred)

# pred = np.mean(pred_en, axis=0)

submit(output_path, id, pred)
 

