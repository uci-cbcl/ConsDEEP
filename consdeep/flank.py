#!/usr/bin/env python

import sys



def main():
    infile = open(sys.argv[1])
    flank = int(sys.argv[2])
    
    for line in infile:
        chrom, start, end = line.strip('\n').split('\t')[0:3]
        start = str(int(start)-flank)
        end = str(int(end)+flank)
        print chrom+'\t'+start+'\t'+end
    
    
    
    
if __name__ == '__main__':
    main()

