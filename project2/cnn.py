from __future__ import print_function
from __future__ import absolute_import

import keras
from keras.models import Sequential, Model, Input
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
import pandas as pd
from scipy.misc import imread, imresize
import numpy as np
import os

from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception


height = 299
width = 299
num_class = 5
batch_size = 64
epochs = 5
data_path = '/Users/lingxiaozhang/Documents/6000b/project2/'

train_path = pd.read_table("train.txt", sep=' ', header=None)
train_path.columns = ['path', 'label']

val_path = pd.read_table("val.txt", sep=' ', header=None)
val_path.columns = ['path', 'label']

test_path = pd.read_table("test.txt", sep=' ', header=None)
test_path.columns = ['path']

length_train = len(train_path)
length_val = len(val_path)
length_test = len(test_path)

train_X = np.zeros((length_train, height, width, 3))
train_Y = train_path['label']
#train_Y = train_Y.values.reshape((-1, 1))
val_X = np.zeros((length_val, height, width, 3))
val_Y = val_path['label']
#val_Y = val_Y.values.reshape((-1, 1))
test_X = np.zeros((length_test, height, width, 3))


print("begin...load data")
for i in range(length_train):
    image = imread(train_path.iloc[i, :].path)
    image = imresize(image, (height, width))
    train_X[i, :, :, :] = image


for j in range(length_val):
    image = imread(val_path.iloc[j, :].path)
    image = imresize(image, (height, width))
    val_X[j, :, :, :] = image


for m in range(length_test):
    image = imread(test_path.iloc[m, :].path)
    image = imresize(image, (height, width))
    test_X[m, :, :, :] = image

print (train_X.shape[1:])

#train_Y = keras.utils.to_categorical(train_Y, 5)
#val_Y = keras.utils.to_categorical(val_Y, 5)'''

'''
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=train_X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class))
model.add(Activation('softmax'))'''

# Binarlize labels
train_Y = keras.utils.to_categorical(train_Y, 5)
val_Y = keras.utils.to_categorical(val_Y, 5)
#test_Y = keras.utils.to_categorical(test_Y, 5)


#base = VGG16(weights='imagenet', include_top=False)
#base = VGG16(weights=None, include_top=False, input_tensor=Input((224, 224, 3)))
#base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input((224, 224, 3)))
base = Xception(include_top=False, weights='imagenet', input_tensor=Input((299, 299, 3)))
output = base.output
output = Flatten()(output)
#output = Dense(512, activation='relu')(output)
output = Dropout(0.6)(output)
output = Dense(5, activation='softmax')(output)

# Assign connections
model = Model(input=base.input, outputs=output)

# Define optimizer
optimizer = keras.optimizers.rmsprop(0.0001, decay=1e-6)
#optimizer = keras.optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer=keras.optimizers.Adadelta()

# Train the model using RMSprop
model.compile(loss='categorical_crossentropy', #sparse_categorical_crossentropy, categorical_crossentropy
              optimizer=optimizer,
              metrics=['accuracy'])


# Normalize
train_X = train_X.astype('float32')
train_X /= 255
val_X = val_X.astype('float32')
val_X /= 255
test_X = test_X.astype('float32')
test_X /= 255

# Feed inputs and labels
print("Start training....")
model.fit(train_X, train_Y,
          batch_size= batch_size,
          epochs=epochs,
          validation_data=(val_X, val_Y)
          )


scores = model.evaluate(val_X, val_Y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


#model_path = os.path.join("./", "model6.h5")
print("Model being saved...")
model.save("m33.h5")

print("Model being loaded...")
model1 = load_model("m33.h5")
pre1 = model1.predict(test_X)
print(pre1)


# Print result in a txt file
result = np.argmax(pre1, axis = 1)
print(result)
print(result.shape)
np.savetxt(data_path + "project2_20475043u2.txt", result, fmt="%d", delimiter=",")



