#!/usr/bin/env python

import sys
import time

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(0)

NB_FILTER = 200
NB_HIDDEN = 200
FILTER_LEN = 5
POOL_FACTOR = 1
DROP_OUT_CNN = 0.1
DROP_OUT_MLP = 0.1
ACTIVATION = 'relu'
BATCH_SIZE = 512
NB_EPOCH = 100
LR = 0.01/2


def main():
    save_name = sys.argv[1]
    nb_filter = int(sys.argv[2])
    nb_hidden = int(sys.argv[3])
    dropout_cnn = float(sys.argv[4])
    dropout_mlp = float(sys.argv[5])
    filter_len = int(sys.argv[6])
    
    print 'loading data...'
    sys.stdout.flush()
    
    X_tr = np.load('X_tr_float32.npy')
    Y_tr = np.load('Y_tr_float32.npy')
    X_va = np.load('X_va_float32.npy')
    Y_va = np.load('Y_va_float32.npy')
    X_te = np.load('X_te_float32.npy')
    Y_te = np.load('Y_te_float32.npy')
    
    __, seq_len, channel_num = X_tr.shape
    pool_len = (seq_len-filter_len+1)/POOL_FACTOR
    
    model = Sequential()
    
    model.add(Convolution1D(input_dim=channel_num,
                        input_length=seq_len,
                        nb_filter=nb_filter,
                        border_mode='valid',
                        filter_length=filter_len,
                        activation=ACTIVATION))
    model.add(MaxPooling1D(pool_length=pool_len, stride=pool_len))
    model.add(Dropout(dropout_cnn))
    model.add(Flatten())
    
    model.add(Dense(nb_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_mlp))
    
    model.add(Dense(input_dim=nb_hidden, output_dim=1))
    model.add(Activation('sigmoid'))

    adagrad = Adagrad(lr=LR)
 
    print 'model compiling...'
    sys.stdout.flush()
     
#    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode='binary')
    model.compile(loss='binary_crossentropy', optimizer=adagrad, class_mode='binary') 
   
    checkpointer = ModelCheckpoint(filepath=save_name+'.hdf5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    outmodel = open(save_name+'.json', 'w')
    outmodel.write(model.to_json())
    outmodel.close()
    
    print 'training...'
    sys.stdout.flush()
    
    time_start = time.time()
    model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, 
              show_accuracy=True, validation_data=(X_va, Y_va),
              callbacks=[checkpointer, earlystopper])
    time_end = time.time()
    
    model.load_weights(save_name+'.hdf5')
#     Y_va_hat = model.predict(X_va, BATCH_SIZE, verbose=1)
#     Y_te_hat = model.predict(X_te, BATCH_SIZE, verbose=1)
    loss_va, acc_va = model.evaluate(X_va, Y_va, show_accuracy=True)
    loss_te, acc_te = model.evaluate(X_te, Y_te, show_accuracy=True)


    print '*'*100
    print '%s accuracy_va : %.4f' % (save_name, acc_va)
    print '%s accuracy_te : %.4f' % (save_name, acc_te)
    print '%s training time : %d sec' % (save_name, time_end-time_start)
    
if __name__ == '__main__':
    main()


