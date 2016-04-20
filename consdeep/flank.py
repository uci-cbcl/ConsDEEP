#!/usr/bin/env python

import sys



def main():
    infile = open(sys.argv[1])
    flank = int(sys.argv[2])
    
    for line in infile:
        fields = line.strip('\n').split('\t')
        chrom, start, end = fields[0:3]
        start = str(int(start)-flank)
        end = str(int(end)+flank)
        print chrom+'\t'+start+'\t'+end+'\t'+'\t'.join(fields[3:])
    
    
    
    
if __name__ == '__main__':
    main()

