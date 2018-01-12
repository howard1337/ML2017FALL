import sys
import numpy as np
from keras.models import load_model
from sklearn.cluster import KMeans

data = np.load(sys.argv[2])
data = data / 255 
model = load_model(sys.argv[1])	
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.summary()

pred = model.predict(data,verbose = 1)
# np.save('prediction.npy',pred)
# pred = np.load('prediction.npy')
# pred = np.load('tsne.npy')
kmeans = KMeans(n_clusters=2, random_state=0).fit(pred)
prediction = kmeans.predict(pred)
cnt = 0
for i in prediction:
	if i == 0:
		cnt += 1
print(cnt)
np.save('cluster.npy',prediction)