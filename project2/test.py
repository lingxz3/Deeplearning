from keras.models import load_model
from scipy.misc import imread, imresize
import pandas as pd
import numpy as np
import keras

height = 299
width = 299
num_class = 5

data_path = '/Users/lingxiaozhang/Documents/6000b/project2/project2/'

#train_path = pd.read_table("train.txt", sep=' ', header=None)
#train_path.columns = ['path', 'label']

val_path = pd.read_table("./val.txt", sep=' ', header=None)
val_path.columns = ['path', 'label']

#length_train = len(train_path)
length_val = len(val_path)


#train_Y = train_path['label']
#train_Y = train_Y.values.reshape((-1, 1))
val_X = np.zeros((length_val, height, width, 3))
val_Y = val_path['label']
#val_Y = val_Y.values.reshape((-1, 1))

test_path = pd.read_table("./test.txt", sep=' ', header=None)
test_path.columns = ['path']
length_test = len(test_path)
test_X = np.zeros((length_test, height, width, 3))


for m in range(length_test):
    image = imread(test_path.iloc[m, :].path)
    image = imresize(image, (height, width))
    test_X[m, :, :, :] = image


#for i in range(length_train):
    #print(train_path.iloc[i, :])
  #  img = imread(train_path.iloc[i, :].path)
   # img = imresize(img, (height, width))
    #train_X[i, :, :, :] = img

for j in range(length_val):
    image = imread(val_path.iloc[j, :].path)
    image = imresize(image, (height, width))
    val_X[j, :, :, :] = image

#print (train_X.shape[1:])

#train_Y = keras.utils.to_categorical(train_Y, 5)
val_Y = keras.utils.to_categorical(val_Y, 5)

val_X = val_X.astype('float32')
val_X /= 255
test_X = test_X.astype('float32')
test_X /= 255

print("load the model...")
model1 = load_model("m33.h5")
#scores = model1.evaluate(train_X, train_Y, verbose=1)
#ste = model1.predict(train_X)
print("evaluate validation set...")
pre = model1.evaluate(val_X, val_Y)
pre1 = model1.predict(test_X)
#print('Train loss:', scores[0])
#print('Train accuracy:', scores[1])
#print(ste)

print('Test loss:', pre[0])
print('Test accuracy:', pre[1])

print(pre1)

print ("Write to txt file...")
result = np.argmax(pre1, axis = 1)
print(result)
print(result.shape)
np.savetxt(data_path + "project2_20475043.txt", result, fmt="%d", delimiter=",")
