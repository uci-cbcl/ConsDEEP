#!/usr/bin/env python

import sys

import pysam


def main():
    inbed = open(sys.argv[1])
    infasta = pysam.Fastafile(sys.argv[2])
    
    for line in inbed:
        chrom, start, end, __, ref, alt = line.strip('\n').split('\t')[0:6]
        start = int(start)
        end = int(end)
        seq = infasta.fetch(chrom, start, end).upper()
        alt_idx = len(seq)/2
        seq = list(seq)
        seq[alt_idx] = alt
        seq = ''.join(seq)
        
        print seq
    
    inbed.close()
    infasta.close()
    
    
    
if __name__ == '__main__':
    main()

