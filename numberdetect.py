import numpy as np
import pandas as pd
from keras.utils import np_utils

np.random.seed(10)

#匯入資料
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print('train data= ',len(x_train_image))
print('test data= ',len(x_test_image))

import matplotlib.pyplot as plt


def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)

    if num>25:num=25

    for i in range(0,num):
        ax=plt.subplot(5,5,i+1)

        ax.imshow(images[idx],cmap='binary')

        title="label="+str(labels[idx])

        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])

        ax.set_title(title,fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1

    plt.show()


plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)



x_train=x_train_image.reshape(60000,784).astype('float32')
x_test=x_test_image.reshape(10000,784).astype('float32')


x_train_normalize=x_train/255
x_test_normalize=x_test/255

y_trainonehot=np_utils.to_categorical(y_train_label)
y_testonehot=np_utils.to_categorical(y_test_label)



from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)

#匯入資料
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print('train data= ',len(x_train_image))
print('test data= ',len(x_test_image))

import matplotlib.pyplot as plt


def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)

    if num>25:num=25

    for i in range(0,num):
        ax=plt.subplot(5,5,i+1)

        ax.imshow(images[idx],cmap='binary')

        title="label="+str(labels[idx])

        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])

        ax.set_title(title,fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1

    plt.show()


plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)



x_train=x_train_image.reshape(60000,784).astype('float32')
x_test=x_test_image.reshape(10000,784).astype('float32')


x_train_normalize=x_train/255
x_test_normalize=x_test/255

y_trainonehot=np_utils.to_categorical(y_train_label)
y_testonehot=np_utils.to_categorical(y_test_label)



from keras.models import Sequential
from keras.layers import Dense
