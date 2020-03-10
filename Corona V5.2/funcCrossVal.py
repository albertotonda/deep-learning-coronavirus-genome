import numpy as np
import math
import random
#Ouput*******************************************************************************
def openMatrix (name) :
	dataArray = np.genfromtxt(name, delimiter=' ')
	data=dataArray
	print(dataArray.shape)
	maxData=np.nanmax(dataArray)
	minData=np.nanmin(dataArray)
	meanData=np.nanmean(dataArray)
	distance=maxData-minData
	print('Max ',maxData)
	print('Min ',minData)
	print('meanData ',meanData)
	for i in range (len(dataArray)):
		for j in range (len(dataArray[0])):
			data[i][j]=(data[i][j]-minData)/distance
			#data[i][j]=data[i][j]
			if math.isnan(data[i][j]):
				data[i][j]=-1
	return data

def openVector (name) :
	dat=np.genfromtxt(name, delimiter=' ')
	print(dat.shape)
	return dat

def saveMatrix(name,var):
	np.savetxt(name, var, fmt='%1.3f', delimiter=' ')

def saveMatrixInt(name,var):
	np.savetxt(name,var, fmt='%i', delimiter=' ')

def saveVectorInt(name,var):
	np.savetxt(name, var, fmt='%i', delimiter=' ')
