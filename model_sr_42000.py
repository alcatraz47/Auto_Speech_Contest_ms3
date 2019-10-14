import librosa
from sklearn.utils import shuffle
import json
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
from tensorflow.python.keras.layers import MaxPooling2D,BatchNormalization
from tensorflow.python.keras.preprocessing import sequence

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def extract_mfcc(data,sr=16000):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d,sr=sr,n_mfcc=24)
        r = r.transpose()
        results.append(r)
    return results

def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='post')

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)

def cnn_model(input_shape,num_class,max_layer_num=2):
        model = Sequential()
        min_size = min(input_shape[:2])
        for i in range(max_layer_num):
            if i == 0:
                model.add(Conv2D(64,3,input_shape = input_shape))
            else:
                model.add(Conv2D(32,3))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            min_size //= 2
            if min_size < 2:
                break
                
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(rate=0.5))
        model.add(Activation('relu'))
        model.add(Dense(num_class))
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

        #extract train feature
        fea_x = extract_mfcc(train_x, sr = 42000)
        max_len = max([len(_) for _ in fea_x])
        fea_x = pad_seq(fea_x, max_len)

        num_class = self.metadata['class_num']
        X=fea_x[:,:,:, np.newaxis]
        y=train_y
        
        model = cnn_model(X.shape[1:],num_class)

        optimizer = tf.keras.optimizers.SGD(lr=0.01,decay=1e-6)
        model.compile(loss = 'sparse_categorical_crossentropy',
                     optimizer = optimizer,
                     metrics= ['accuracy'])
        model.summary()
        # callbacks = [tf.keras.callbacks.EarlyStopping(
        #             monitor='val_loss', patience=10)]
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.00001)]
        history = model.fit(X,ohe2cat(y),
                    epochs=100,
                    callbacks=callbacks,
                    validation_split=0.1,
                    verbose=1,  # Logs once per epoch.
                    batch_size=32,
                    shuffle=True)

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
        fea_x = extract_mfcc(test_x, 42000)
        fea_x = pad_seq(fea_x, max_len)
        test_x=fea_x[:,:,:, np.newaxis]

        #predict
        y_pred = model.predict_classes(test_x)

        test_num=self.metadata['test_num']
        class_num=self.metadata['class_num']
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(y_pred):
            y_test[idx][y] = 1

        return y_test
