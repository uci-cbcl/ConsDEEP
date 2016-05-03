#!/usr/bin/env python

import sys
import time

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, Flatten, Masking
from keras.layers.recurrent import LSTM
from keras.optimizers import Adagrad, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(0)

NB_FILTER = 200
NB_HIDDEN = 200
FILTER_LEN = 10
POOL_FACTOR = 1
DROP_OUT_CNN = 0.1
DROP_OUT_MLP = 0.1
ACTIVATION = 'relu'
BATCH_SIZE = 512/2
NB_EPOCH = 100
LR = 0.01


def main():
    base_name = sys.argv[1]
    save_name = sys.argv[2]
    nb_lstm = int(sys.argv[3])
    nb_hidden = int(sys.argv[4])
    dropout_lstm = float(sys.argv[5])
    dropout_mlp = float(sys.argv[6])
    
    print 'loading data...'
    sys.stdout.flush()
    
    X_tr = np.load('X_'+base_name+'_tr_float32.npy')
    Y_tr = np.load('Y_'+base_name+'_tr_float32.npy')
    X_va = np.load('X_'+base_name+'_va_float32.npy')
    Y_va = np.load('Y_'+base_name+'_va_float32.npy')
    X_te = np.load('X_'+base_name+'_te_float32.npy')
    Y_te = np.load('Y_'+base_name+'_te_float32.npy')
    
    n_tr, seq_len, channel_num = X_tr.shape
    n_va, seq_len, channel_num = X_va.shape
    n_te, seq_len, channel_num = X_te.shape    
        
    model = Sequential()

    model.add(Masking(mask_value=0.0, input_shape=(seq_len, channel_num)))
    model.add(LSTM(nb_lstm, input_dim=channel_num, input_length=seq_len))
    model.add(Dropout(dropout_lstm))
    model.add(Dense(nb_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_mlp))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    rmsprop = RMSprop(lr=LR)
 
    print 'model compiling...'
    sys.stdout.flush()
     
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
   
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


