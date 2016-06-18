#!/usr/bin/env python

import sys

import numpy as np


def main():
    base_name = sys.argv[1]
    save_name = sys.argv[2]
    
    X_tr = np.load('X_' + base_name + '_tr.npy')
    X_va = np.load('X_' + base_name + '_va.npy')
    X_te = np.load('X_' + base_name + '_te.npy')
    
    X_mean = X_tr.mean(axis=0)
    X_std = X_tr.std(axis=0) + 1e-5
    
    X_tr = (X_tr - X_mean)/X_std
    X_va = (X_va - X_mean)/X_std
    X_te = (X_te - X_mean)/X_std
    
    np.save('X_' + save_name + '_tr.npy', X_tr)
    np.save('X_' + save_name + '_va.npy', X_va)
    np.save('X_' + save_name + '_te.npy', X_te)
    
    
    
if __name__ == '__main__':
    main()

