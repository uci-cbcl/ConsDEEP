#!/usr/bin/env python

import sys

MEME_HEADER = """MEME version 4.4

ALPHABET= ACGT

strands: + -

Background letter frequencies (from web form):
A 0.25000 C 0.25000 G 0.25000 T 0.25000 
"""

BASE_PWM = [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]

BASE_DICT = {'A':0, 'C':1, 'G':2, 'T':3}

def main():
    infile = open(sys.argv[1])
    text = infile.read()
    
    header_lst = map(lambda x:x.split('\n')[0], text.split('>')[1:])
    seq_lst = map(lambda x:''.join(x.split('\n')[1:]), text.split('>')[1:])
    N = len(header_lst)
    
    print MEME_HEADER
    for i in range(0, N):
        print '\nMOTIF %s\n' % (header_lst[i])
        print 'letter-probability matrix: alength= 4'
        
        for j in range(0, len(seq_lst[i])):
            print '%f\t%f\t%f\t%f' % tuple(BASE_PWM[BASE_DICT[seq_lst[i][j]]])
        
    




    infile.close()
    
    
if __name__ == '__main__':
    main()