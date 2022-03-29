#!/usr/bin/env python3
import numpy as np
import fileinput
import sys
import argparse

#***********************************************************************
# Parsing arguments
#***********************************************************************
parser = argparse.ArgumentParser(description='Convert white into alpha of a column of hex colors of format "0xRRGGBB". Outputs "0xAARRGGBB".')
parser.add_argument('files', type=str, nargs='*', help='Files with colors to be converted.')
args = parser.parse_args()
#***********************************************************************
# reading data as argument or std input
data = np.loadtxt(fileinput.input(args.files), dtype=float)
#***********************************************************************



# printing the result to std out
#np.savetxt(sys.stdout, np.stack(datanew,axis=1), newline='\n')

