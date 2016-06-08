#!/usr/bin/env python

import sys

import numpy as np
import theano as th
from keras.models import model_from_json, Sequential
from keras.layers.core import Masking
from keras.layers.recurrent import LSTM


BATCH_SIZE = 512


def main():
    X = np.load(sys.argv[1])
    base_name = sys.argv[2]
    save_name = sys.argv[3]
    
    base_json = base_name+'.json'
    base_hdf5 = base_name+'.hdf5'
    
    model = model_from_json(open(base_json).read())
    model.load_weights(base_hdf5)
    
    N, seq_len, channel_num = X.shape
    nb_lstm = model.layers[1].output_dim

    model_ = Sequential()
    model_.add(Masking(mask_value=0.0, input_shape=(seq_len, channel_num)))
    model_.add(LSTM(nb_lstm, input_dim=channel_num, input_length=seq_len, return_sequences=True))
    model_.layers[1].set_weights(model.layers[1].get_weights())
    
    f = th.function([model_.get_input()], model_.get_output())
    
    states = np.zeros((N, seq_len, nb_lstm), dtype='float32')
    
    i = 0
    while i+BATCH_SIZE < N:
        states_i = f(X[i:i+BATCH_SIZE])
        states[i:i+BATCH_SIZE, :, :] = states_i[:, :, :]
        i += BATCH_SIZE
        print '%s/%s' % (i, N)
        sys.stdout.flush()
        
    states_i = f(X[i:N])
    states[i:N, :, :] = states_i[:, :, :]
    
    np.save(save_name, states)
    
    
    
    
    
if __name__ == '__main__':
    main()