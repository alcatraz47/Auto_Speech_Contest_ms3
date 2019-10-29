import os
import pickle
import pandas as pd
import time

import librosa
from sklearn.utils import shuffle
import json
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, Input
from keras.layers import *
from keras.layers import MaxPooling2D,BatchNormalization
from keras.preprocessing import sequence
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def extract_melspectrogram_train(data, sr=42000):
    X_train = []
    for feature in data:
        # melspectrogram = librosa.feature.melspectrogram(feature, sr = sr, n_mels = 40).transpose()
        melspectrogram = librosa.feature.mfcc(feature, sr = sr, n_mfcc = 40).transpose()
        X_train.append(melspectrogram)

    max_len = max([len(_) for _ in X_train])
    X_train = pad_seq(X_train, max_len)
    X_train = np.asarray(X_train)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train_normalized = (X_train - mean) / std
    

    return X_train_normalized, max_len

def extract_melspectrogram_test(data, max_seq, sr = 42000):
    results = []
    for d in data:
        # r = librosa.feature.melspectrogram(d,sr=16000,n_mels=40).transpose()
        r = librosa.feature.mfcc(d, sr = sr, n_mfcc = 40).transpose()
        results.append(r)
    results = pad_seq(results, max_seq)
    results = np.asarray(results)
    # print(results.shape)
    return results

def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='post')

def extract_mfcc_test(data,sr=16000):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=40)
        # r = r.transpose()
        results.append(r)
    results = np.asarray(results)
    return results

def test_y_pre(data_path):
    dataset = pd.read_csv(data_path, header=None)
    arr = np.asarray(dataset)
    index = []

    for data in arr:
        string = data[0]
        string  = string.replace(' ', '')
        i  = string.find('1')
        index.append(i)
    index = np.asarray(index)
    index = keras.utils.to_categorical(index)

    return index

class Resnet32:
    @staticmethod
    def resnet(data, num_filters, stride, reduce_dimension, reg):
        shortcut = data

        #bn -> ac -> conv2d(stride = (1,1))
        bn_1 = BatchNormalization()(data)
        ac_1 = Activation("relu")(bn_1)

        conv_1 = Conv2D(int(num_filters * .25), (1, 1), padding = 'same', use_bias = False, kernel_regularizer = l2(reg))(ac_1)

        #bn -> ac -> conv2d(stride = (3,3))
        bn_2 = BatchNormalization()(conv_1)
        ac_2 = Activation("relu")(bn_2)

        conv_2 = Conv2D(int(num_filters * .25), (3, 3), padding = 'same', use_bias = False, kernel_regularizer = l2(reg))(ac_2)

        #bn -> ac -> conv2d(stride = (1, 1))
        bn_3 = BatchNormalization()(conv_2)
        ac_3 = Activation("relu")(bn_3)

        conv_3 = Conv2D(num_filters, (1, 1), padding = 'same', use_bias = False, kernel_regularizer = l2(reg))(ac_3)

        #for spatial size reducing
        if reduce_dimension:
            shortcut = Conv2D(num_filters, (1, 1), use_bias = False, kernel_regularizer = l2(reg))(ac_1)

        #final conv layer adding
        resnet = add([conv_3, shortcut])
        return resnet

    @staticmethod
    def build_model(X_train, classes, stages, num_filters, reg):

        inputs = Input(shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = BatchNormalization()(inputs)

        #conv2d(stride = (5, 5)) -> bn -> act -> pool
        x = Conv2D(num_filters[0], (5, 5), use_bias = False, padding = "same", kernel_regularizer = l2(reg))(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)

        #stacking up resnets
        for i in range(0, len(stages)):
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            x = Resnet32.resnet(x, num_filters[i+1], stride, reduce_dimension = True, reg = reg)

            for j in range(0, stages[i] - 1):
                if j%2 == 0:
                    x = Resnet32.resnet(x, num_filters[i+1], (1, 1), reduce_dimension = True, reg = reg)
                else:
                    x = Resnet32.resnet(x, num_filters[i+1], (1, 1), reduce_dimension = False, reg = reg)


        #avoid ffc and use average pooling
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D(8, 8)(x)

        #softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = keras.models.Model(inputs, x)
        return model

class Lstm:
    def lstm_model(self, input_shape, class_num):

        # lstm_input = Input(shape = (40,32))
        lstm_input = Input(shape = input_shape)

        lstm = CuDNNLSTM(32, return_sequences = True, kernel_regularizer=l2(0.01), kernel_initializer = 'random_normal')(lstm_input)
        activate = Activation("relu")(lstm)
        batch_norm = BatchNormalization()(activate)
        lstm = CuDNNLSTM(32, return_sequences = True, kernel_regularizer=l2(0.01), kernel_initializer = 'random_normal')(batch_norm)
        activate = Activation("relu")(lstm)
        batch_norm = BatchNormalization()(activate)
        lstm = CuDNNLSTM(16, return_sequences = True, kernel_regularizer=l2(0.01), kernel_initializer = 'random_normal')(batch_norm)
        activate = Activation("relu")(lstm)
        batch_norm = BatchNormalization()(activate)
        lstm = CuDNNLSTM(16, return_sequences = True, kernel_regularizer=l2(0.01), kernel_initializer = 'random_normal')(batch_norm)
        activate = Activation("relu")(lstm)
        batch_norm = BatchNormalization()(activate)
        lstm = CuDNNLSTM(10, return_sequences = True, kernel_regularizer=l2(0.01), kernel_initializer = 'random_normal')(batch_norm)
        activate = Activation("relu")(lstm)
        batch_norm = BatchNormalization()(activate)
        lstm = CuDNNLSTM(10, kernel_regularizer=l2(0.01), kernel_initializer = 'random_normal')(batch_norm)
        activate = Activation("relu")(lstm)
        batch_norm = BatchNormalization()(activate)
        dense = Dense(100, activation = 'relu', kernel_regularizer=l2(0.01))(batch_norm)
        dense = Dropout(.2)(dense)
        batch_norm = BatchNormalization()(dense)
        dense = Dense(100, activation = 'relu', kernel_regularizer=l2(0.01))(batch_norm)
        dense = Dropout(.2)(dense)
        batch_norm = BatchNormalization()(lstm)
        dense = Dense(100, activation = 'relu', kernel_regularizer=l2(0.01))(batch_norm)
        dense = Dropout(.2)(dense)
        dense = Dense(class_num, activation = 'softmax')(dense)

        model = keras.models.Model(lstm_input , dense)

        return model

class Model(object):

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.
        
        :param train_dataset: tuple, (x_train, y_train)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        t1 = time.time()
        if self.done_training:
            return
        train_x, train_y = train_dataset
        train_x, train_y = shuffle(train_x, train_y)

        # data = zip(train_x, train_y)

        X_train, max_len = extract_melspectrogram_train(train_x)
        y_train = np.asarray(train_y)

        num_class = self.metadata['class_num']
        X_train = X_train[:,:,:, np. newaxis]

        stages = (3, 4, 6)
        resnet = Resnet32()
        print(resnet)
        num_filters = (16, 32, 64, 128)
        model = resnet.build_model( X_train,num_class, stages, num_filters, float(0.0001))
        # input_shape = (X_train.shape[1], X_train.shape[2])
        # lstm_object = Lstm()
        # model = lstm_object.lstm_model(input_shape, num_class)


        # PATH = 'sample_data/DEMO/'
        # y_test = test_y_pre(os.path.join(PATH , 'data01.solution'))



        # PATH_X_TEST = 'sample_data/DEMO/data01.data'
        # pickle_in = open(os.path.join(PATH_X_TEST, 'test.pkl'), "rb")
        # x_test = pickle.load(pickle_in)
        # x_test = np.asarray(x_test)
        # x_test = extract_melspectrogram_test(x_test, max_len)

        # print("----------------------------------------------------eikhane-------------------------------------------")
        # print(x_test.shape)


        # optimizer = tf.keras.optimizers.SGD(lr=0.01,decay=1e-6)
        optimizer = keras.optimizers.Adam(lr = 0.0001)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr = 0.00001)
        
        model.compile(loss = 'categorical_crossentropy',
                     optimizer = optimizer,
                     metrics= ['accuracy']) #sparse_
        model.summary()
        with open(self.train_output_path + '/feature.config', 'wb') as f:
            f.write(str(max_len).encode())
            f.close()
            
        for epoch in range(100):
            t2 = time.time()
            if(round(t2 - t1) >= 1680):
                break
            history = model.fit(X_train,y_train, epochs=1, callbacks=[reduce_lr], validation_split=0.1, verbose=1, batch_size=32, shuffle=True)

            model.save(self.train_output_path + '/model.h5')
        # history = model.fit(X_train,y_train,
        #             epochs=200,
        #             verbose=1,
        #             callbacks = [reduce_lr],
        #             validation_split = 0.1,
        #             batch_size=32, 
        #             shuffle=True)#validation_data = (x_test, y_test),

        # model.save(self.train_output_path + '/model.h5')


        self.done_training=True

    def test(self, test_x, remaining_time_budget=None):
        """
        :param x_test: list of vectors, input test speech raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        model = keras.models.load_model(self.test_input_path + '/model.h5')
        with open(self.test_input_path + '/feature.config', 'r') as f:
            max_len = int(f.read().strip())
            f.close()

        #extract test feature
        fea_x = extract_melspectrogram_test(test_x,max_len)
        # test_x = fea_x
        test_x = fea_x[:,:,:, np.newaxis]



        #predict
        y_pred = model.predict(test_x)
        y_pred = np.argmax(y_pred, axis = 1)

        test_num=self.metadata['test_num']
        class_num=self.metadata['class_num']
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(y_pred):
            y_test[idx][y] = 1

        return y_test