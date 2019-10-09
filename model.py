import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import librosa
from sklearn.utils import shuffle
import json
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
from keras.layers import MaxPooling2D,BatchNormalization
from keras.preprocessing import sequence
# from tensorflow.python.keras import models
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
# from tensorflow.python.keras.layers import MaxPooling2D,BatchNormalization
# from tensorflow.python.keras.preprocessing import sequence

from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def extract_melspectrogram_train(data, sr=16000):
    X_train, X_val, y_train, y_val = [], [], [], []
    counter = 1
    for feature, label in data:
        melspectrogram = librosa.feature.melspectrogram(feature, sr = sr, n_mels = 40)#.transpose()
        if(counter > 10):
            X_val.append(melspectrogram)
            y_val.append(label)
            counter = 1
            continue
        X_train.append(melspectrogram)
        y_train.append(label)
        counter += 1

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train_normalized = (X_train - mean) / std
    X_val_normalized = (X_val - mean) / std

    return X_train_normalized, X_val_normalized, y_train, y_val

def extract_melspectrogram_test(data, sr = 16000):
    results = []
    for d in data:
        r = librosa.feature.melspectrogram(d,sr=16000,n_mels=40)
        results.append(r)
    results = np.asarray(results)
    print(results.shape)
    return results

# def convert_to_categorical(data):
#     encoder = LabelEncoder()
#     encoded_y = encoder.fit_transform(data)
#     encoded_data = to_categorical(encoded_y)
#     return encoded_data

def extract_mfcc_train(data, sr = 16000):
    X_train, X_val, y_train, y_val = [], [], [], []
    counter = 1
    for feature, label in data:
        mfcc = librosa.feature.mfcc(feature, sr = sr, n_mfcc = 40)#.transpose()
        if(counter > 10):
            X_val.append(mfcc)
            y_val.append(label)
            counter = 1
            continue
        X_train.append(mfcc)
        y_train.append(label)
        counter += 1

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train_normalized = (X_train - mean) / std
    X_val_normalized = (X_val - mean) / std

    return X_train_normalized, X_val_normalized, y_train, y_val

def extract_mfcc_test(data,sr=16000):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=40)
        # r = r.transpose()
        results.append(r)
    results = np.asarray(results)
    return results

def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='post')

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)

def cnn_model(input_shape,num_class,max_layer_num=5):
        # model = Sequential()
        # min_size = min(input_shape[:2])
        # for i in range(max_layer_num):
        #     if i == 0:
        #         model.add(Conv2D(64,3,input_shape = input_shape,padding='same'))
        #     else:
        #         model.add(Conv2D(64,3,padding='same'))
        #     model.add(Activation('relu'))
        #     model.add(BatchNormalization())
        #     model.add(MaxPooling2D(pool_size=(2,2)))
        #     min_size //= 2
        #     if min_size < 2:
        #         break
                
        # model.add(Flatten())
        # model.add(Dense(64))
        # model.add(Dropout(rate=0.5))
        # model.add(Activation('relu'))
        # model.add(Dense(num_class))
        # model.add(Activation('softmax'))
        model = Sequential()
        model.add(Conv2D(30, (3,3), input_shape = input_shape, padding = 'same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(25, (3,3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(20, (3,3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(num_class))
        model.add(Dropout(.2))
        model.add(Activation('softmax'))

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
        if self.done_training:
            return
        train_x, train_y = train_dataset
        train_x, train_y = shuffle(train_x, train_y)
        max_len = max([len(_) for _ in train_x])
        train_x = pad_seq(train_x, max_len)

        data = zip(train_x, train_y)

        #extract train feature
        X_train, X_val, y_train, y_val = extract_mfcc_train(data)
        # max_len = max([len(_) for _ in fea_x])
        # fea_x = pad_seq(fea_x, max_len)
        # X_train, X_val, y_train, y_val = extract_melspectrogram_train(data)
        # X_val = pad_seq(X_val, max_len)

        num_class = self.metadata['class_num']
        X_train = X_train[:,:,:, np. newaxis]
        # print('-------------------------------------------------eikhane----------------------------')
        # print(X_train.shape)
        X_val = X_val[:,:,:, np.newaxis]
        y_train = np.asarray(y_train)
        # y_train = convert_to_categorical(y_train)
        # print(y_train.shape)
        

        y_val = np.asarray(y_val)
        # y_val = convert_to_categorical(y_val)
        print(y_val.shape)
        
        model = cnn_model(X_train.shape[1:],num_class)

        # optimizer = tf.keras.optimizers.SGD(lr=0.01,decay=1e-6)
        optimizer = keras.optimizers.Adam(lr = 0.0001)
        model.compile(loss = 'categorical_crossentropy',
                     optimizer = optimizer,
                     metrics= ['accuracy']) #sparse_
        model.summary()
        # callbacks = [tf.keras.callbacks.EarlyStopping(
        #             monitor='val_loss', patience=10)]
        callbacks = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001)
        history = model.fit(X_train,y_train,
                    epochs=1000,
                    callbacks = [callbacks],
                    validation_data = (X_val, y_val),
                    verbose=1,  # Logs once per epoch.
                    batch_size=40, 
                    shuffle=True)#, shuffle=True, validation_split=0.1 #ohe2cat #callbacks=callbacks, validation_data = (X_val, y_val)

        model.save(self.train_output_path + '/model.h5')

        with open(self.train_output_path + '/feature.config', 'wb') as f:
            f.write(str(max_len).encode())
            f.close()

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
        model = models.load_model(self.test_input_path + '/model.h5')
        with open(self.test_input_path + '/feature.config', 'r') as f:
            max_len = int(f.read().strip())
            f.close()

        #extract test feature
        # fea_x = extract_mfcc(test_x)
        # fea_x = pad_seq(fea_x, max_len)
        # test_x=fea_x[:,:,:, np.newaxis]

        # max_shape = max([len(_) for _ in test_x])
        test_x = pad_seq(test_x, max_len)
        # fea_x = extract_melspectrogram_test(test_x)
        fea_x = extract_mfcc_test(test_x)
        
        test_x = fea_x[:,:,:, np.newaxis]



        #predict
        y_pred = model.predict_classes(test_x)

        test_num=self.metadata['test_num']
        class_num=self.metadata['class_num']
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(y_pred):
            y_test[idx][y] = 1

        return y_test

