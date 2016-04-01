#!/usr/bin/env python

import sys

import numpy as np

from consdeep.utils import onehot


def main():
    infile = open(sys.argv[1])
    base_name = sys.argv[2]
    seq_len_max = int(sys.argv[3])
    
    lines = map(lambda x:x.strip('\n'), infile.readlines())
    N = len(lines)
    X = np.zeros((N, seq_len_max, 4))
    Y = np.zeros((N, 1))
    
    for i in range(0, N):
        seq, label = lines[i].split('\t')
        seq_len = len(seq)
        x = onehot(seq)
        y = int(label)       
        X[i, 0:seq_len, :] = x
        Y[i, :] = y
        
        if i%10000 == 0:
            print '%s/%s lines processed...' % (i, N)
    
    
    np.save('X_%s_float32.npy' % (base_name), X.astype('float32'))
    np.save('Y_%s_float32.npy' % (base_name), Y.astype('float32'))
    
    
    infile.close()
    
if __name__ == '__main__':
    main()


