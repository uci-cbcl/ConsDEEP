#!/usr/bin/env python

import sys



def main():
    infile = open(sys.argv[1])
    pad = int(sys.argv[2])
    
    flank = pad/2
    
    for line in infile:
        fields = line.strip('\n').split('\t')
        chrom, start, end = fields[0:3]
        start_new = str(int(start)-flank)
        end_new = str(int(start)+flank)
        print chrom+'\t'+start_new+'\t'+end_new+'\t'+'\t'.join(fields[3:])
    
    
    
    
if __name__ == '__main__':
    main()

