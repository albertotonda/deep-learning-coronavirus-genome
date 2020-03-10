# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
# this is an incredibly useful function
from pandas import read_csv

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./best/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./best/labels.csv", header=None)
		
	return dfData.as_matrix(), dfLabels.as_matrix().ravel() # to have it in the format that the classifiers like


def runFeatureReduce() :

	labels=5
	cMatrix=np.zeros((labels, labels))
	
	for i in range (0,int(sys.argv[1])):
		#results0_500_128_128_128_128_4_2_2_4_2_2.txt
		#0_500_130_204_150_196_148_236_81_9_106_121
		dfLabels = read_csv("./values/"+"test"+str(i)+"_1000_130_204_150_196_148_236_81_9_106_121.txt", header=None)
		y_test=dfLabels.as_matrix().ravel()
		
		dfLabels = read_csv("./values/"+"results"+str(i)+"_1000_130_204_150_196_148_236_81_9_106_121.txt", header=None)
		y_new=dfLabels.as_matrix().ravel()
		
		for j in range(0,len(y_new)):
				cMatrix[y_test[j]][y_new[j]]+=1

	pd.DataFrame(cMatrix).to_csv("./cMatrix.csv", header=None, index =None)
	
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )