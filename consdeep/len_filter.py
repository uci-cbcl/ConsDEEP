#!/usr/bin/env python

import sys



def main():
    infile = open(sys.argv[1])
    len_min = int(sys.argv[2])
    len_max = int(sys.argv[3])
    
    for line in infile:
        chrom, start, end = line.strip('\n').split('\t')[0:3]
        length = int(end)-int(start)
        
        if length >= len_min and length <= len_max:
            print chrom+'\t'+start+'\t'+end
    
    
    
    
if __name__ == '__main__':
    main()

