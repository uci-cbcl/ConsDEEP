#!/usr/bin/env python

import sys

import numpy as np
import theano as th
from keras.models import model_from_json

BATCH_SIZE = 512


def main():
    X = np.load(sys.argv[1])
    Y = np.load(sys.argv[2])
    base_name = sys.argv[3]
    
    base_json = base_name+'.json'
    base_hdf5 = base_name+'.hdf5'
    base_meme = base_name+'.meme'
    
    model = model_from_json(open(base_json).read())
    model.load_weights(base_hdf5)
    W, b = model.layers[0].get_weights()
    nb_filter, channel_num, filter_len, _ = W.shape
    N, seq_len, channel_num = X.shape
    loss_base, acc_base = model.evaluate(X, Y, batch_size=BATCH_SIZE, show_accuracy=True)
    
    for i in range(0, nb_filter):
        W_i = np.array(W)*0
        b_i = np.array(b)*0
        W_i[i, :, :, :] = W[i, :, :, :]
        b_i[i] = b[i]
        
        model.layers[0].set_weights((W_i, b_i))
        loss_i, acc_i = model.evaluate(X, Y, batch_size=BATCH_SIZE, show_accuracy=True)
        
        print 'activate_FILTER_LEN%s_%s_acc/all_acc : %.2f%%/%.2f%%' % (filter_len, i, acc_i*100, acc_base*100)
        
        
    
    
    
    
    
    
if __name__ == '__main__':
    main()


