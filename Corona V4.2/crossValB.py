# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
import math
import random
import sys
from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from funcCNN import *

from pandas import read_csv
import pandas as pd 
from sklearn.model_selection import StratifiedKFold

def get_InfoA(indexVar,labelSize,vectorSize):
	
	trainLabels=openVector('trainLabels.txt')
	validLabels=openVector('validateLabels.txt')
	testLabels=openVector('testLabels.txt')
	
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)
	
	trainArray = np.genfromtxt('./'+'Train.matrix', delimiter=' ')
	train=np.array(trainArray)
	print('train set', train.shape)
	
	testArray = np.genfromtxt('./'+'Test.matrix', delimiter=' ')
	test=np.array(testArray)
	print('test set', test.shape)
	
	validateArray = np.genfromtxt('./'+'Validate.matrix', delimiter=' ')
	valid=np.array(validateArray)
	print('valid set', valid.shape)

	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels)

from sklearn.preprocessing import StandardScaler
def get_Info(indexVar,labelSize,vectorSize):
	#examples=8129

	data=[]
	data = np.genfromtxt('./data/'+'data.csv', delimiter=',')
	data=np.array(data)
	print('data set', data.shape)
	#print(data[0])
	
	from sklearn import preprocessing
	StandardScaler = preprocessing.StandardScaler()
	data = StandardScaler.fit_transform(data)

	
	labels=openVector('./data/labels.csv')
	print(labels)
	#print(labels)
	testIndex=openVector('./data/index/'+str(indexVar)+'test_index.txt')
	valIndex=openVector('./data/index/'+str(indexVar)+'val_index.txt')
	trainIndex=openVector('./data/index/'+str(indexVar)+'train_index.txt')
	
	testIndex=testIndex.astype(int)
	valIndex=valIndex.astype(int)
	trainIndex=trainIndex.astype(int)
		
	train=[]
	test=[]
	valid=[]
	
	
	trainLabels=[]
	testLabels=[]
	validLabels=[]
	#test***************************************************************************
	for i in range (0,len(testIndex)):
		testLabels.append(labels[testIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[testIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[testIndex[i]][j])
		test.append(temp)
	#valid***************************************************************************
	for i in range (0,len(valIndex)):
		validLabels.append(labels[valIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[valIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[valIndex[i]][j])
		valid.append(temp)
	#train***************************************************************************
	for i in range (0,len(trainIndex)):
		trainLabels.append(labels[trainIndex[i]])
		temp=[]
		for j in range (0,len(data[0])):
			if(data[trainIndex[i]][j]==-1):
				temp.append(0)
			else:
				temp.append(data[trainIndex[i]][j])
		train.append(temp)
	
	test=np.array(test)
	testLabels=np.array(testLabels)

	valid=np.array(valid)
	validLabels=np.array(validLabels)
	
	train=np.array(train)
	trainLabels=np.array(trainLabels)
	
	
	
	print(train.shape)
	print(trainLabels.shape)
	print(valid.shape)
	print(validLabels.shape)
	print(test.shape)
	print(testLabels.shape)
	
	print(labelSize)
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)


	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels)


