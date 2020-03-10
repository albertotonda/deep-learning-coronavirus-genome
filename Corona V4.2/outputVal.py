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

def ouputVal(oneHot_test_labels,results,filter_width,kfoldIndex,limit,var,test_accuracy,valid_accuracy,iterCurve,validCurve,trainingCurve):
	fpr=[]
	tpr=[]
	tresholds=[]
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, tresholds = roc_curve(oneHot_test_labels[:, 1], results[:, 1])
	roc_auc = auc(fpr, tpr)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'ROC.png')
	plt.clf()

	import matplotlib as mpl2
	mpl2.use('Agg')
	import matplotlib.pyplot as plt2
	plt2.title('Accuracy')
	plt2.plot(iterCurve, trainingCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt2.xlabel('Iterations')
	plt2.ylabel('Training Accuracy')
	plt2.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'Training_Accuracy.png')
	plt2.clf()

	import matplotlib as mpl3
	mpl3.use('Agg')
	import matplotlib.pyplot as plt3
	plt3.title('Accuracy')
	plt3.plot(iterCurve, validCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt3.xlabel('Iterations')
	plt3.ylabel('Valid Accuracy')
	plt3.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'Valid_Accuracy.png')
	plt3.clf()

	saveMatrix('results_'+str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'%1.4f_'%(roc_auc)+'%1.4f_'%(valid_accuracy)+'%1.4f'%(test_accuracy),results)
	
def ouputValA(oneHot_test_labels,results,filter_width,kfoldIndex,limit,var,test_accuracy,valid_accuracy,iterCurve,validCurve,trainingCurve,cross_entropyCurve):
	fpr=[]
	tpr=[]
	tresholds=[]
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, tresholds = roc_curve(oneHot_test_labels[:, 1], results[:, 1])
	roc_auc = auc(fpr, tpr)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'ROC.png')
	plt.clf()

	import matplotlib as mpl2
	mpl2.use('Agg')
	import matplotlib.pyplot as plt2
	plt2.title('Accuracy')
	plt2.plot(iterCurve, trainingCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt2.xlabel('Iterations')
	plt2.ylabel('Training Accuracy')
	plt2.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'Training_Accuracy.png')
	plt2.clf()



	import matplotlib as mpl3
	mpl3.use('Agg')
	import matplotlib.pyplot as plt3
	plt3.title('Accuracy')
	plt3.plot(iterCurve, validCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt3.xlabel('Iterations')
	plt3.ylabel('Valid Accuracy')
	plt3.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'Valid_Accuracy.png')
	plt3.clf()

	saveMatrix('results_'+str(filter_width)+'_'+str(kfoldIndex)+'_'+str(var)
	+'_'+'%1.2f_'%(limit)+'%1.4f_'%(roc_auc)+'%1.4f_'%(valid_accuracy)+'%1.4f'%(test_accuracy),results)

def ouputValIter(oneHot_test_labels,results,filter_width,kfoldIndex,
				 limit,iterMax,test_accuracy,valid_accuracy,iterCurve,
				 validCurve,trainingCurve,typeProgram):
	fpr=[]
	tpr=[]
	tresholds=[]
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, tresholds = roc_curve(oneHot_test_labels[:, 1], results[:, 1])
	roc_auc = auc(fpr, tpr)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'ROC.png')
	plt.clf()

	import matplotlib as mpl2
	mpl2.use('Agg')
	import matplotlib.pyplot as plt2
	plt2.title('Accuracy')
	plt2.plot(iterCurve, trainingCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt2.xlabel('Iterations')
	plt2.ylabel('Training Accuracy')
	plt2.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'Training_Accuracy.png')
	plt2.clf()

	import matplotlib as mpl3
	mpl3.use('Agg')
	import matplotlib.pyplot as plt3
	plt3.title('Accuracy')
	plt3.plot(iterCurve, validCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt3.xlabel('Iterations')
	plt3.ylabel('Valid Accuracy')
	plt3.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'Valid_Accuracy.png')
	plt3.clf()

	saveMatrix('results_'+str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'%1.4f_'%(roc_auc)+'%1.4f_'%(valid_accuracy)+'%1.4f'%(test_accuracy),results)
	
def ouputValIterA(oneHot_test_labels,results,filter_width,kfoldIndex,
				 limit,iterMax,test_accuracy,valid_accuracy,iterCurve,
				 validCurve,trainingCurve,typeProgram,cross_entropyCurve):
	fpr=[]
	tpr=[]
	tresholds=[]
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, tresholds = roc_curve(oneHot_test_labels[:, 1], results[:, 1])
	roc_auc = auc(fpr, tpr)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'ROC.png')
	plt.clf()

	import matplotlib as mpl2
	mpl2.use('Agg')
	import matplotlib.pyplot as plt2
	plt2.title('Accuracy')
	plt2.plot(iterCurve, trainingCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt2.xlabel('Iterations')
	plt2.ylabel('Training Accuracy')
	plt2.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'Training_Accuracy.png')
	plt2.clf()

	import matplotlib as mpl3
	mpl3.use('Agg')
	import matplotlib.pyplot as plt3
	plt3.title('Accuracy')
	plt3.plot(iterCurve, validCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt3.xlabel('Iterations')
	plt3.ylabel('Valid Accuracy')
	plt3.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'Valid_Accuracy.png')
	plt3.clf()
	
	import matplotlib as mpl4
	mpl4.use('Agg')
	import matplotlib.pyplot as plt4
	plt4.title('Accuracy')
	plt4.plot(iterCurve, cross_entropyCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt4.xlabel('Iterations')
	plt4.ylabel('cross_entropy')
	plt4.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'cross_entropy.png')
	plt4.clf()

	saveMatrix('results_'+str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'%1.4f_'%(roc_auc)+'%1.4f_'%(valid_accuracy)+'%1.4f'%(test_accuracy),results)
	
def ouputValIterA(oneHot_test_labels,results,filter_width,kfoldIndex,
				 limit,iterMax,test_accuracy,valid_accuracy,iterCurve,
				 validCurve,trainingCurve,typeProgram,cross_entropyCurve,cross_entropyCurveT):
	fpr=[]
	tpr=[]
	tresholds=[]
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, tresholds = roc_curve(oneHot_test_labels[:, 1], results[:, 1])
	roc_auc = auc(fpr, tpr)

	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'ROC.png')
	plt.clf()

	import matplotlib as mpl2
	mpl2.use('Agg')
	import matplotlib.pyplot as plt2
	plt2.title('Accuracy Training')
	plt2.plot(iterCurve, trainingCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt2.xlabel('Iterations')
	plt2.ylabel('Training Accuracy')
	plt2.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'Training_Accuracy.png')
	plt2.clf()

	import matplotlib as mpl3
	mpl3.use('Agg')
	import matplotlib.pyplot as plt3
	plt3.title('Accuracy Validation')
	plt3.plot(iterCurve, validCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt3.xlabel('Iterations')
	plt3.ylabel('Valid Accuracy')
	plt3.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'Valid_Accuracy.png')
	plt3.clf()
	
	import matplotlib as mpl4
	mpl4.use('Agg')
	import matplotlib.pyplot as plt4
	plt4.title('cross_entropy Validation')
	plt4.plot(iterCurve, cross_entropyCurve, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt4.xlabel('Iterations')
	plt4.ylabel('cross_entropy')
	plt4.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'cross_entropy.png')
	plt4.clf()

	import matplotlib as mpl5
	mpl5.use('Agg')
	import matplotlib.pyplot as plt5
	plt5.title('cross_entropy Training')
	plt5.plot(iterCurve, cross_entropyCurveT, 'r')
	#plt.plot([0,1],[0,1],'r--')
	#plt.xlim([-0.1,1.2])
	#plt.ylim([-0.1,1.2])
	plt5.xlabel('Iterations')
	plt5.ylabel('cross_entropyT')
	plt5.savefig(str(filter_width)+'_'+str(kfoldIndex)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'cross_entropyT.png')
	plt5.clf()

	saveMatrix('results_'+str(filter_width)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+typeProgram
	+'_'+'%1.2f_'%(limit)+'%1.4f_'%(roc_auc)+'%1.4f_'%(valid_accuracy)+'%1.4f'%(test_accuracy),results)