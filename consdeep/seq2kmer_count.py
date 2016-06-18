#!/usr/bin/env python

import sys

import numpy as np


def get_kmer(K):
    kmer_lst = [['A', 'C', 'G', 'T']]

    if K == 1:
        return kmer_lst[0]

    for k in range(1, K):
        kmer_lst.append([])
        
        for kmer in kmer_lst[k-1]:
            kmer_lst[k].append(kmer + 'A')
            kmer_lst[k].append(kmer + 'C')
            kmer_lst[k].append(kmer + 'G')
            kmer_lst[k].append(kmer + 'T')
    
    kmer_lst_combined = []
    
    for k in range(0, K):
        kmer_lst_combined.extend(kmer_lst[k])
    
    return kmer_lst_combined
    


def main():
    infile = open(sys.argv[1])
    base_name = sys.argv[2]
    K = int(sys.argv[3])
    
    kmer_lst = get_kmer(K)
    D = len(kmer_lst)
        
    lines = map(lambda x:x.strip('\n'), infile.readlines())
    N = len(lines)
    X = np.zeros((N, D))
    
    for i in range(0, N):
        seq, label = lines[i].split('\t')
       
        X[i] = map(lambda x:seq.count(x), kmer_lst)
        
        if i%10000 == 0:
            print '%s/%s lines processed...' % (i, N)
            sys.stdout.flush()
    
    np.save('X_%s.npy' % (base_name), X)
    
    
    infile.close()
    
if __name__ == '__main__':
    main()


