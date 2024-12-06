import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, GlobalAveragePooling2D, ELU
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Cropping2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

import time, cv2, glob, os

global inputShape,size

def kerasModel4():
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size,size,1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())
        # model.add(Dropout(.2))
        # model.add(Activation('relu'))
        # model.add(Dense(1024))
        # model.add(Dropout(.5))
        model.add(Dense(512))
        model.add(Dropout(.1))
        model.add(Activation('relu'))
        # model.add(Dense(256))
        # model.add(Dropout(.5))
        # model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model

size=100

 ## load Training data : pothole
potholeTrainImages = glob.glob(os.path.join(os.getcwd(), "My Dataset/train/Pothole/*.jpg"))
potholeTrainImages.extend(glob.glob(os.path.join(os.getcwd(), "My Dataset/train/Pothole/*.jpeg")))
potholeTrainImages.extend(glob.glob(os.path.join(os.getcwd(), "My Dataset/train/Pothole/*.png")))

train1 = [cv2.imread(img,0) for img in potholeTrainImages]
for i in range(0,len(train1)):
    train1[i] = cv2.resize(train1[i],(size,size))
temp1 = np.asarray(train1)


#  ## load Training data : non-pothole
nonPotholeTrainImages = glob.glob(os.path.join(os.getcwd(), "My Dataset/train/Plain/*.jpg"))
# nonPotholeTrainImages.extend(glob.glob("My Dataset/train/Plain/*.jpeg"))
# nonPotholeTrainImages.extend(glob.glob("My Dataset/train/Plain/*.png"))
train2 = [cv2.imread(img,0) for img in nonPotholeTrainImages]
# train2[train2 != np.array(None)]
for i in range(0,len(train2)):
    train2[i] = cv2.resize(train2[i],(size,size))
temp2 = np.asarray(train2)



## load Testing data : non-pothole
nonPotholeTestImages = glob.glob(os.path.join(os.getcwd(), "My Dataset/test/Plain/*.jpg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.jpeg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.png"))
test2 = [cv2.imread(img,0) for img in nonPotholeTestImages]
# train2[train2 != np.array(None)]
for i in range(0,len(test2)):
    test2[i] = cv2.resize(test2[i],(size,size))
temp4 = np.asarray(test2)


## load Testing data : potholes
potholeTestImages = glob.glob(os.path.join(os.getcwd(), "My Dataset/test/Pothole/*.jpg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.jpeg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.png"))
test1 = [cv2.imread(img,0) for img in potholeTestImages]
# train2[train2 != np.array(None)]
for i in range(0,len(test1)):
    test1[i] = cv2.resize(test1[i],(size,size))
temp3 = np.asarray(test1)


X_train = []
X_train.extend(temp1)
X_train.extend(temp2)
X_train = np.asarray(X_train)

X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)





y_train1 = np.ones([temp1.shape[0]],dtype = int)
y_train2 = np.zeros([temp2.shape[0]],dtype = int)
y_test1 = np.ones([temp3.shape[0]],dtype = int)
y_test2 = np.zeros([temp4.shape[0]],dtype = int)

print(y_train1[0])
print(y_train2[0])
print(y_test1[0])
print(y_test2[0])

y_train = []
y_train.extend(y_train1)
y_train.extend(y_train2)
y_train = np.asarray(y_train)

y_test = []
y_test.extend(y_test1)
y_test.extend(y_test2)
y_test = np.asarray(y_test)


X_train,y_train = shuffle(X_train,y_train)
X_test,y_test = shuffle(X_test,y_test)

# X_train.reshape([-1,50,50,1])
# X_test.reshape([-1,50,50,1])/
X_train = X_train.reshape(X_train.shape[0], size, size, 1)
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (size, size, 1)
model = kerasModel4()


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=50,
                    epochs=100,
                    verbose=1,
                    validation_split=0.2,
                    shuffle=True)

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

print("Saving model weights and configuration file")

# Save weights
model.save_weights('new_model.weights.h5')

# Save model architecture
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)