#!/usr/bin/env python

import sys

import pysam


def main():
    inbed = open(sys.argv[1])
    infasta = pysam.Fastafile(sys.argv[2])
    
    for line in inbed:
        chrom, start, end = line.strip('\n').split('\t')[0:3]
        start = int(start)
        end = int(end)
        seq = infasta.fetch(chrom, start, end).upper()
        
        print '>'+':'.join(line.strip('\n').split('\t'))
        print seq

        
        
    
    
    
if __name__ == '__main__':
    main()

