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
	fig1, ax1 = plt.subplots()
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	yTest=[]
	yNew=[]
	
	for i in range (0,10):
		#results0_500_128_128_128_128_4_2_2_4_2_2.txt
		dfLabels = read_csv("./values/"+"test"+str(i)+"_1000_130_204_150_196_148_236_81_9_106_121.txt", header=None)
		y_test=dfLabels.as_matrix().ravel()
		for j in range(0,len(y_test)):
			if (y_test[j]>1):
				y_test[j]=1
		
		dfLabels = read_csv("./values/"+"results"+str(i)+"_1000_130_204_150_196_148_236_81_9_106_121.txt", header=None)
		y_new=dfLabels.as_matrix().ravel()
		for j in range(0,len(y_new)):
			if (y_new[j]>1):
				y_new[j]=1
		
		fpr, tpr, thresholds = roc_curve(y_test, y_new)
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		
		ax1.plot(fpr, tpr, lw=1, alpha=0.3,
				 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

	ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
		 label='Chance', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	ax1.plot(mean_fpr, mean_tpr, color='b',
			 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
			 lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					 label=r'$\pm$ 1 std. dev.')

	ax1.axis(xmin=-0.05,xmax=1.05)
	ax1.axis(ymin=-0.05,ymax=1.05)

	ax1.set_xlabel('False Positive Rate')
	ax1.set_ylabel('True Positive Rate')
	ax1.set_title('Receiver operating characteristic ')
	ax1.legend(loc="lower right")
	plt.show()
	
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )