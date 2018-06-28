
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
import numpy as np


# In[5]:


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[6]:


fig = plt.figure(figsize=(20,20))
for i in range(6):
    ax = fig.add_subplot(1,6, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(str(y_train[i]))


# ## Preproccessing and data exploration
# One-hot encode y_train og y_test. 
# Set features to between 0-1 by dividing with 255

# In[7]:


from keras.utils import to_categorical
np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
input_shape = x_train.shape
print(input_shape)
print(y_train[0:4])
#print(x_train[0])

y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
print(y_train[1])

x_train = x_train.astype("float32") / 255
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_test = x_test.astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

input_shape = x_train[0].shape
print(x_train[0].shape)
print(x_train[0])


# ## Network
# 
# 

# In[63]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=2, padding='same', activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[64]:


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])


# In[65]:


model.fit(x_train, y_train, batch_size=1000, epochs=30, verbose=1, validation_data=(x_test, y_test))

