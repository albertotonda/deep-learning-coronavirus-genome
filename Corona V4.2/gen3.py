#Declarations******************************************************************************
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
from crossValB import *
from outputVal import *

f=open('outputVector.txt', 'w')
f.write('1\n')
f.write('1\n')
temp=1.0
trueAcc=str(temp)
print(trueAcc)
f.write(trueAcc+'\n')
f.close()
#Parameters*******************************************************************************
#number of classes
labelSize=5
#fixed size of the data
vectorSize=31029
#Maximum number of iterations
iterMax=int(sys.argv[1])
#maximum number of iterations
limit=1.00
#regularization on the weights
beta=0.001
#version of the code
version='gen1'
#size of batch
batchSize=50
#Parameters*******************************************************************************
w1=int(sys.argv[2]) #32
w2=int(sys.argv[3]) #64
w3=int(sys.argv[4]) #128
w4=int(sys.argv[5]) #256
h1=int(sys.argv[6]) #4
h2=int(sys.argv[7]) #4
h3=int(sys.argv[8]) #4
wd1=int(sys.argv[9]) #4
wd2=int(sys.argv[10]) #4
wd3=int(sys.argv[11]) #4
index=int(sys.argv[12]) #0
kfoldIndex=int(sys.argv[13]) #0
generation=int(sys.argv[14]) #0
#Model declaration************************************************************************
import tensorflow as tf
#declare interactive session
sess = tf.InteractiveSession()

#INPUT->CONV LAYER->CONV LAYER->CONV LAYER->RECT FLAT->RECT DROPOUT

#function to declare easily the weights only by shape
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
#function to declare easily the bias only by shape
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#input variable
x = tf.placeholder(tf.float32, [None, vectorSize])
#keep probability to change from dropout 0.50 to 1.0 in validation and test
keep_prob = tf.placeholder(tf.float32)
#expected outputs variable
y_ = tf.placeholder(tf.float32, [None, labelSize])

#arrange the tensor as an image (8*545) 1 channel
x_image0 = tf.reshape(x, [-1,1,vectorSize,1])
x_image = tf.transpose(x_image0, perm=[0,3,2,1])
#arrange the tensor into 8 channels (1*545) 8 channels

#1 LAYER*************************************************************************************
#1 Convolutional Layer Explicit for regularization of the weights
#weigth first layer 8 input channels, 32 output channels, 1x4 filter window size
W_conv1 = weight_variable([1, wd1, 1, w1])
#bias declaration the size has to be the same as the output channels (32)
b_conv1 = bias_variable([w1])
#convolution (input weights) moving 1 step each time with a relu
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, 
	strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
#max pooling with a 4 width window size, moving 4 in width by step
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, h1, 1],
	strides=[1, 1, h1, 1], padding='SAME')
#output=545/4
#1 LAYER*************************************************************************************

#2 LAYER*************************************************************************************
#2 Convolutional Layer Explicit for regularization of the weights
#weigth first layer 32 input channels, 64 output channels, 1x4 filter window size
W_conv2 = weight_variable([1, wd2, w1, w2])
b_conv2 = bias_variable([w2])
#convolution (input weights) moving 1 step each time with a relu
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, 
	strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
#max pooling with a 4 width window size, moving 4 in width by step
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, h2, 1],
	strides=[1, 1, h2, 1], padding='SAME')
#output=545/16
#2 LAYER*************************************************************************************

#3 LAYER*************************************************************************************
#3 Convolutional Layer Explicit for regularization of the weights
#weigth first layer 64 input channels, 128 output channels, 1x4 filter window size
W_conv3 = weight_variable([1, wd3, w2, w3])
b_conv3 = bias_variable([w3])
#convolution (input weights) moving 1 step each time with a relu
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, 
	strides=[1, 1,1, 1], padding='SAME') + b_conv3)
#max pooling with a 4 width window size, moving 4 in width by step
h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 1, h3, 1],
	strides=[1, 1, h3, 1], padding='SAME')
#output=545/64
#3 LAYER*************************************************************************************

#Rectifier LAYER*****************************************************************************
#calculated coefficient for the flattening from the size of the 3 convolutional layer
coef=int (h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])
h_pool2_flat = tf.reshape(h_pool3, [-1, coef])
#declare the weights considering the constants and 256 output 
W_fc1 = weight_variable([coef, w4])
b_fc1 = bias_variable([w4])

#rectifier (matmul)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#Rectifier LAYER*****************************************************************************

#Rectifier-Dropout LAYER**********************************************************************
#dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#declare weights with the ouput layer in this case 2 (labelSize)
W_fc2 = weight_variable([w4, labelSize])
b_fc2 = bias_variable([labelSize])
#output
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#Rectifier-Dropout LAYER**********************************************************************

#Loss Function********************************************************************************
#cross entropy (loss function) mean value input with actual input and add the regularized 
#weights by the beta value (0.01) for the 3 layers
#tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying t
#he softmax function (but it does it all together in a more mathematically careful way). 
#It's similar to the result of:
#sm = tf.nn.softmax(x)
#ce = cross_entropy(sm)
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[0]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)+
	beta*tf.nn.l2_loss(W_conv1)+
	beta*tf.nn.l2_loss(W_conv2)+
	beta*tf.nn.l2_loss(W_conv3))
#Optimizer Adam at 1e-4 (literature)**********************************************************
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
#softmax prediction remember we are using one hot labels
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

trueResult=tf.argmax(y_conv,1)
trueTest=tf.argmax(y_,1)
#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Loss Function********************************************************************************
valid_accuracy_global=0.0
test_accuracy_global=0.0


#start
sess.run(tf.initialize_all_variables())

#Extra to verify sizes************************************************************************
print(x_image.get_shape())
#print(h_conv1.get_shape())
print(h_pool1.get_shape())
#print(h_conv2.get_shape())
print(h_pool2.get_shape())
#print(h_conv3.get_shape())
print(h_pool3.get_shape())

print(y_conv.get_shape())
print(y_.get_shape())
#Extra to verify sizes************************************************************************

#Input Data***********************************************************************************
(test_dataset_Flat,oneHot_test_labels,valid_dataset_Flat,oneHot_valid_labels,
	train_dataset_Flat,oneHot_train_labels)=get_Info(
	kfoldIndex,labelSize,vectorSize)
runs=int(len(oneHot_train_labels)/batchSize)
print(runs)
#Output:

yResult=[]
yTest=[]
#Ouput Data Variables*************************************************************************
#name to save the results (unused,fold,version,iterations)
#name=str(str(kfoldIndex)+'_'+version+'_'+str(iterMax)+'_'+str(w1)+'_'+str(w2)+'_'+str(w3)+'_'+
#		 str(w4)+'_'+str(h1)+'_'+str(h2)+'_'+str(h3)+'_'+str(wd1)+'_'+str(wd2)+'_'+str(wd3))
name=str(str(kfoldIndex)+'_'+str(iterMax)+'_'+str(w1)+'_'+str(w2)+'_'+str(w3)+'_'+
		 str(w4)+'_'+str(h1)+'_'+str(h2)+'_'+str(h3)+'_'+str(wd1)+'_'+str(wd2)+'_'+str(wd3))
f = open('./values/'+name+'.txt', 'a')
#Ouput Data Variables*************************************************************************

#Main Loop************************************************************************************
#initialize variables
iter=0
train_accuracy=0.0
valid_accuracy=0.0
test_accuracy=0.0
#best validation accuracy
best=0
validWindow=[0,0,0]
repeatWindow=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#limit (normally a dummy value 0.80) and iterations...
while ((best<limit) & (iter<iterMax)):
	indexBatch=[]
	for iB in range (0,len(oneHot_train_labels)):
		indexBatch.append(iB)
	random.shuffle(indexBatch)
	for run in range (0,runs):
		#Get data from train set and store in xa (inputs),ya(labels),ca(constants) 100 batch size
		xa,ya=getBatch_run(train_dataset_Flat,oneHot_train_labels,batchSize,run,indexBatch)
		#pass the values and 0.50 dropout (literature)
		train_step.run(feed_dict={x: xa, y_: ya, keep_prob: 0.5})
		#print each 10 iterations
	#calculate train accuracy
	xa,ya=getBatch(train_dataset_Flat,oneHot_train_labels,batchSize)
	train_accuracy = accuracy.eval(feed_dict={
		x:xa, y_: ya, keep_prob: 1.0})
	#calculate validation accuracy
	valid_accuracy=accuracy.eval(feed_dict={
		x: valid_dataset_Flat, y_: oneHot_valid_labels, keep_prob: 1.0})
	#calculate validation loss
	cross_entropyVal=cross_entropy.eval(feed_dict={
		x: valid_dataset_Flat, y_: oneHot_valid_labels, keep_prob: 1.0})
	#calculate train loss
	cross_entropyTrain=cross_entropy.eval(feed_dict={
		x: xa, y_: ya, keep_prob: 1.0})
	#append values for graphs
	#if valid accuracy is better than the best accuracy then calculate test accuracy
	#if valid_accuracy>best:
	validWindowValue=0
	tempValid=validWindow
	for i in range(0,len(validWindow)-1):
		tempValid[i]=validWindow[i+1]
	for i in range(0,len(validWindow)):
		validWindow[i]=tempValid[i]
	validWindow[len(validWindow)-1]=valid_accuracy
	for i in range(0,len(validWindow)):
		validWindowValue=validWindowValue+validWindow[i]
	validWindowValue=validWindowValue/len(validWindow)
	tempValid=repeatWindow
	for i in range(0,len(repeatWindow)-1):
		tempValid[i]=repeatWindow[i+1]
	for i in range(0,len(repeatWindow)):
		repeatWindow[i]=tempValid[i]
	repeatWindow[len(repeatWindow)-1]=valid_accuracy
	if np.var(repeatWindow)==0 and iter>10:
		iter=iter
	if (validWindowValue)>best:
		best=(validWindowValue)
		#calculate test accuracy
		test_accuracy= accuracy.eval(feed_dict={x:test_dataset_Flat, 
			y_: oneHot_test_labels, keep_prob: 1.0})
		#calculate the results of the whole model, probabilities in one hot format
		results=correct_prediction.eval(feed_dict={x:test_dataset_Flat, y_: oneHot_test_labels, keep_prob: 1.0})
		yResult=trueResult.eval(feed_dict={x:test_dataset_Flat, y_: oneHot_test_labels, keep_prob: 1.0})
		yTest=trueTest.eval(feed_dict={x:test_dataset_Flat, y_: oneHot_test_labels, keep_prob: 1.0})
		fOut=open('outputVector.txt', 'w')
		fOut.write('1\n')
		fOut.write('1\n')
		temp=1.0-best
		trueAcc=str(temp)
		print(trueAcc)
		fOut.write(trueAcc+'\n')
		fOut.close()				
	#append everything to a log for retrieving results
	log="%d	%d	%g	%g	%g	%g	%g	%g"%(iter,kfoldIndex,train_accuracy,valid_accuracy,best,
		test_accuracy,cross_entropyVal,cross_entropyTrain)
	print(log)
	f.write(log+'\n')
	iter=iter+1
#Main Loop************************************************************************************
#calculate validation accuracy
#valid_accuracy=accuracy.eval(feed_dict={
#	x: valid_dataset_Flat, y_: oneHot_valid_labels, keep_prob: 1.0})
#print("valid accuracy %g"%valid_accuracy)

#calculate test accuracy from the whole train data set
#print("test accuracy %g"%accuracy.eval(feed_dict={x: test_dataset_Flat,
#	y_: oneHot_test_labels, keep_prob: 1.0}))
#close log file
f.close()
saveVectorInt('./values/results'+name+'.txt',yResult)
saveVectorInt('./values/test'+name+'.txt',yTest)
f = open('log3.txt', 'a')
name=str(str(index)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+str(test_accuracy)+'_'+str(valid_accuracy)+'_'+str(best)+'_'+str(w1)+'_'+str(w2)
		 +'_'+str(w3)+'_'+ str(w4)+'_'+str(h1)+'_'+str(h2)+'_'+str(h3)+'_'+str(wd1)+'_'
		 +str(wd2)+'_'+str(wd3)+'_'+str(generation))
f.write(name+'\n')
f.close()

f=open(str(index)+'.index','a')
name=str(str(index)+'_'+str(kfoldIndex)+'_'+str(iterMax)+'_'+str(test_accuracy)+'_'+str(valid_accuracy)+'_'+str(best)+'_'+str(w1)+'_'+str(w2)
		 +'_'+str(w3)+'_'+ str(w4)+'_'+str(h1)+'_'+str(h2)+'_'+str(h3)+'_'+str(wd1)+'_'
		 +str(wd2)+'_'+str(wd3)+'_'+str(generation))
f.write(name+'\n')
f.close()
#Input Data***********************************************************************************
#close session
f=open('outputVector.txt', 'w')
f.write('1\n')
f.write('1\n')
temp=1.0-best
trueAcc=str(temp)
print(trueAcc)
f.write(trueAcc+'\n')
f.close()

sess.close()


































