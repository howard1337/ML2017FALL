import os
import sys
from skimage import io
from skimage.transform import rescale
import numpy as np
from numpy.linalg import eig

filename = sys.argv[1]
face = []
my_list = os.listdir(filename)
for i in my_list:
	face.append(io.imread(filename + '/' + str(i)).flatten())
	print(filename + '/' + str(i))
face = np.array(face).T

print(face.shape)
print('load image complete')
mean = face.mean(axis=1)
face = (face.T - mean).T



U, s, V = np.linalg.svd(face, full_matrices=False)
# np.save('U.npy',U)
# np.save('s.npy',s)
# np.save('V.npy',V)
# sys.exit()

# U = np.load('U.npy')
# s = np.load('s.npy')
# V = np.load('V.npy')
#0 25 333 129

target = io.imread(filename + '/' + sys.argv[2]).flatten()
target = (target - mean) 
weight = np.dot(U[:,:4].T,target)
# weight = np.absolute(weight)
reconstruct = np.zeros(shape = face[:,0].shape)
# ratio = weight / np.sum(weight)
print(weight)
# print(U)
# temp = np.round(U[:,1])
# print(temp.shape)
# io.imshow(temp.reshape(600,600,3))
# io.show()
# sys.exit()


for i in range(4):
	reconstruct += ((weight[i] * U[:,i]))
reconstruct += mean
reconstruct -= np.min(reconstruct)
reconstruct /= np.max(reconstruct)
reconstruct = (reconstruct * 255).astype(np.uint8)
# reconstruct = np.round(reconstruct)
# reconstruct *= -1
# io.imshow((ttt.reshape(600,600,3)))
io.imsave('reconstruction.jpg',reconstruct.reshape(600,600,3),quality = 100)


# sum_value = np.sum(s)
# print(s[0] / sum_value,s[1] / sum_value,s[2] / sum_value, s[3] / sum_value)

