#!/usr/bin/env python

import sys
import os

import numpy as np
import theano as th
import theano.tensor as T
from keras.models import model_from_json
from matplotlib import pyplot as plt

LR = 10
LR_MIN = 1e-3
DELTA = 1e-4
SEQ_TRUNC = 90
SEQ_LEN = 1000
ITER_MAX= 200

def main():
    base_name = sys.argv[1]
    save_folder = sys.argv[2]
    
    base_json = base_name+'.json'
    base_hdf5 = base_name+'.hdf5'
    model = model_from_json(open(base_json).read())
    model.load_weights(base_hdf5)
    
    channel_num, nb_lstm = model.layers[1].get_weights()[0].shape
    
    x = model.get_input()
    y = T.arctanh(model.layers[1].get_output())
    
    X_init = np.random.normal(0.25, 0.01, (2, SEQ_LEN, 4)).astype('float32')
    X_init[1, :, :] = X_init[0, :, :]
    X_init/X_init.sum(axis=2).reshape((2, SEQ_LEN, 1))

    try:
        os.mkdir(save_folder)
    except:
        pass

    for idx in range(0, nb_lstm):
        g = T.grad(T.mean(y[:, idx]), x)
        f_g = th.function([x], g)
        f_y = th.function([x], T.mean(y[:, idx]))
        
        X_old = np.array(X_init)
        Y_old = -1e5
        lr = LR
        iter = 1
        delta = 1
        
        while lr > LR_MIN and iter <= ITER_MAX and delta >= DELTA:
            X_new = X_old + lr*f_g(X_old)
            
            if np.isnan(X_new.mean()):
                break
            
            X_new = np.maximum(0, X_new)
            X_new = X_new/X_new.sum(axis=2).reshape((2, SEQ_LEN, 1))
            Y_new = f_y(X_new).tolist()

            if np.isnan(Y_new) or (np.isfinite(Y_new) == False):
                break

            if Y_new < Y_old:
                lr = lr/2.0
                continue

            delta = Y_new - Y_old
            
            print 'cell_%s\titer:%s\tdelta:%.5f\tlr:%.3f\tobj:%.5f' % (idx, iter, delta, lr, Y_new)
            sys.stdout.flush()
            
            X_old = np.array(X_new)
            Y_old = Y_new
            iter += 1
        
        outtransfac = open(save_folder + '/cell_%s.transfac' %(idx), 'w')
        outtransfac.write('\t'.join(['P0', 'A', 'C', 'G', 'T']) + '\n')
        
        for j in range(0, SEQ_TRUNC)[::-1]:
            outtransfac.write('0' + str(SEQ_TRUNC-j) + '\t' + '%f\t%f\t%f\t%f\n' % tuple(X_old[0, SEQ_LEN-1-j, :].tolist()))
        
        outtransfac.close()
    
        
    
    
    
    
    
if __name__ == '__main__':
    main()


