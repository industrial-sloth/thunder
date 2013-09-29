# ref <master> <inputFile> <inds> <outputFile>
# 
# compute summary statistics on an xyz stack
# each row is (x,y,z,timeseries)
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 4:
  print >> sys.stderr, \
  "(ref) usage: ref <master> <inputFile> <outputFile> <mode>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	return ((int(vec[0]),int(vec[1]),int(vec[2])),ts) # (x,y,z),(tseries) pair 

# parse inputs
sc = SparkContext(sys.argv[1], "ref")
inputFile = str(sys.argv[2])
outputFile = str(sys.argv[3])
mode = str(sys.argv[4])
logging.basicConfig(filename=outputFile+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(ref) loading data")
lines = sc.textFile(inputFile)
X = lines.map(parseVector)

# get z ordering
logging.info("(ref) getting z ordering")
zinds = X.filter(lambda (k,x) : (k[0] == 1000) & (k[1] == 1000)).map(lambda (k,x) : k[2]).collect()
savemat(outputFile+"zinds.mat",mdict={'zinds':zinds},oned_as='column',do_compression='true')

# compute ref
logging.info('(ref) computing reference image')
if mode == 'med':
	ref = X.map(lambda (k,x) : median(x)).collect()
if mode == 'mean':
	ref = X.map(lambda (k,x) : mean(x)).collect()
if mode == 'std':
	ref = X.map(lambda (k,x) : std((x - median(x))/(median(x)+0.1))).collect()
logging.info('(ref) saving results...')
savemat(outputFile+mode+".mat",mdict={'ref':ref},oned_as='column',do_compression='true')