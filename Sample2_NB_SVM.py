# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:24:38 2018

@author: alexc
"""

import os
import subprocess 
from nltk.util import ngrams
from Analysis import Evaluation
import collections 
import numpy as np

# Naive Bayes implementation for sentiment analysis (Evaluation class at the end)
class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]
        self.posReviews = []
        self.negReviews = []
        self.pred_vec = []

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety 
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        
        """
        # Reset vocabulary
        self.vocabulary = set()
        #Extract vocabulary
        self.extractVocabulary(reviews)
        #Reset condProb and prior
        self.condProb = {}
        self.prior = {}
        # Compute prior probability of positive class
        def computePrior(reviews):
            countPos = 0
            countNeg = 0
            posReviews = []
            negReviews = []
            for review in reviews:
                # Count number of times POS occurs
                if review[0] is 'POS':
                    countPos += 1
                    for token in review[1]:
                        #Append the token to a list where all the words from POS reviwes are stored
                        posReviews.append(token)
                # Count number of times NEG occurs
                elif review[0] is 'NEG':
                    countNeg += 1
                    for token in review[1]:
                   #Append the token to a list where all the words from NEG reviwes are stored
                        negReviews.append(token)
            return (float(countPos)/len(reviews),float(countNeg)/len(reviews),posReviews,negReviews)
        #set smoothing constat 
        if self.smoothing is True:    
            smooth_const = 1.0
        elif self.smoothing is False:
            smooth_const = 0.0
        # Compute prior probabilities 
        posPrior,negPrior,posReviews,negReviews = computePrior(reviews)
        #Count how many times each token occurs in the POS class
        countPositives = collections.Counter(posReviews)
        # Count total number of words occurring in POS clas
        countNegatives = collections.Counter(negReviews)
        for review in reviews:
            for token in review[1]:
                self.condProb[token] =[]
                self.condProb[token].append(np.log((smooth_const+countPositives[token])/(len(self.vocabulary)*smooth_const+len(posReviews))))
                self.condProb[token].append(np.log((smooth_const+countNegatives[token])/(len(self.vocabulary)*smooth_const+len(negReviews))))
        self.prior['POS'] = posPrior
        self.prior['NEG'] = negPrior

        
    def test(self,reviews):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.
        
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        labels = []
        predictedlabels = []
        self.predictions = []
        for review in reviews:
            scorePos = 0
            scoreNeg = 0
            if review[0] is 'POS':
                labels.append('POS')
            elif review[0] is 'NEG':
                labels.append('NEG')
            for token in review[1]:
                if token in self.vocabulary:
                    scorePos += self.condProb[token][0]
                    scoreNeg += self.condProb[token][1]

            # Calculate posterior probabilities over each class
            posPosterior = np.log(self.prior['POS']) + scorePos
            negPosterior = np.log(self.prior['NEG']) + scoreNeg
            # Classify according to the highest 
            if posPosterior >= negPosterior:
                    predictedlabels.append("POS")
            elif posPosterior < negPosterior:
                    predictedlabels.append("NEG")
        predictions = []
        for pair in zip(predictedlabels,labels) :
            if pair[0] == pair[1]:
                predictions.append('+')
            else:   
                predictions.append('-')
        self.predictions = predictions
      #  print "SVM: %.2f" % self.getAccuracy()
     
# Support vector machine for sentiment analysis: implementation relies on SVM light (http://svmlight.joachims.org/)
class SVM(Evaluation):
    """
    general svm class to be extended by text-based classifiers.
    """
    def __init__(self,svmlight_dir,optimise):
        self.predictions=[]
        self.svmlight_dir=svmlight_dir
        self.c = []
        self.sigma = []
        self.r = []
        self.s = []
        self.std_devs = []
        self.accuracies = []
        self.optimise = optimise
    def writeFeatureFile(self,data,filename):
        """
        write local file in svmlight data format.
        see http://svmlight.joachims.org/ for description of data format.

        @param data: input data
        @type data: list of (string, list) tuples where string is the label and list are features in (id, value) tuples

        @param filename: name of file to write
        @type filename: string
        """
        # Remove file if it already exists to avoid adding entries to an existing file
        try:
            os.remove(filename)
        except OSError:
            pass
        
        with open(filename,'a') as out_file:
            for review in data:
                line = []
                line.append(str(review[0])+" ")
                for pair in review[1]:
                    line.append(str(pair[0])+":"+str(pair[1])+" ")
                line.append("\n")
                out_file.writelines(line)
        

    def train(self,train_data):
        """
        train svm 

        @param train_data: training data 
        @type train_data: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set. to be implemented by child 
        self.getFeatures(train_data)
        # function to find vectors (feature, value pairs). to be implemented by child
        train_vectors=self.getVectors(train_data)
        self.writeFeatureFile(train_vectors,"train.data")
        if self.optimise is False:
            # train SVM model with default parameters
            print('Starting Training (No Optimisation):')
            train_process = subprocess.Popen(["svm_learn","train.data","svm_model"],stdout=subprocess.PIPE)
            print (train_process.communicate()[0])
        elif self.optimise is True:
            if not self.sigma and not self.s and not self.r:
            # Optimise C only if no sigma value is provided
                print('Starting Training (Optimisation). Value of c is '+str(self.c))
                train_process = subprocess.Popen(["svm_learn","-c",str(self.c),"train.data","svm_model"+str(self.c)],stdout=subprocess.PIPE)
                print (train_process.communicate()[0])
            if self.sigma:
            # Optimise jointly C and sigma
                print('Starting Training (Optimisation). Value of c is '+str(self.c)+" and value of sigma is"+str(self.sigma))
                train_process = subprocess.Popen(["svm_learn","-c",str(self.c),"-t","2","-g",str(self.sigma),"train.data","svm_model_c"+str(self.c)+"sigma_"+str(self.sigma)],stdout=subprocess.PIPE)
                print (train_process.communicate()[0])
            if self.s and self.r:
                # Optimise SVM quadratic kernel 
                print('Starting Training (Optimisation). Value of s is '+str(self.s)+" and value of r is"+str(self.r))
                print('Value of C is'+str(self.c))
                train_process = subprocess.Popen(["svm_learn","-c",str(self.c),"-t","1","-s",str(self.s),"-r",str(self.r),"train.data","svm_model_s"+str(self.s)+"r_"+str(self.r)],stdout=subprocess.PIPE)
                print (train_process.communicate()[0])
                
    def test(self,test_data):
        """

        @param test_data: test data 
        @type test_data: list of (string, list) tuples corresponding to (label, content)
        """
        # Put the test vectors in SVM format
        test_vectors = self.getVectors(test_data)
        self.writeFeatureFile(test_vectors,"test.data")
        if self.optimise is False:  
            print ('Start testing process (No Optimisation)')
            test_process = subprocess.Popen(["svm_classify","test.data","svm_model","svm_out"],stdout=subprocess.PIPE)
            print (test_process.communicate()[0])
        if self.optimise is True:
            if not self.sigma and not self.s and not self.r:
                # Testing SVM with linear kernel
                print ('Start testing process (Optimisation). Value of c is '+str(self.c))
                test_process = subprocess.Popen(["svm_classify","test.data","svm_model"+str(self.c),"svm_out"],stdout=subprocess.PIPE)
                print( test_process.communicate()[0])
            if self.sigma:
                # Testing SVM with Gaussian kernel
                print ('Start testing process (Optimisation). Value of c is '+str(self.c)+'and value of sigma is '+str(self.sigma))
                test_process = subprocess.Popen(["svm_classify","test.data","svm_model_c"+str(self.c)+"sigma_"+str(self.sigma),"svm_out"],stdout=subprocess.PIPE)
                print (test_process.communicate()[0])
            if self.s and self.r:
                # Testing  SVM quadratic kernel 
                print('Starting Testing (Optimisation). Value of s is '+str(self.s)+" and value of r is"+str(self.r))
                test_process = subprocess.Popen(["svm_classify","test.data","svm_model_s"+str(self.s)+"r_"+str(self.r),"svm_out"],stdout=subprocess.PIPE)
                print (test_process.communicate()[0])
                
        # See model predictions
        scores = []
        svm_predictions = []
        predictions = []
        test_labels = []
        # Read output file 
        with open ("svm_out",'r') as svm_out:
            raw_out = svm_out.readlines()
        scores = [float(label) for label in raw_out]
        for score in scores:
            if score > 0:
                svm_predictions.append("POS")
            elif score <0:
                svm_predictions.append("NEG")
        # Extract labels of testing data
        for review in test_data:
            if review[0] is 'POS':
                test_labels.append('POS')
            elif review[0] is 1:
                test_labels.append('POS')
            elif review[0] is 'NEG':
                test_labels.append('NEG')
            elif review[0] is -1:
                test_labels.append('NEG')
            
        # Compare the two sets of labels
        for pair in zip(svm_predictions,test_labels) :
            if pair[0] == pair[1]:
                predictions.append('+')
            else:   
                predictions.append('-')
        self.predictions = predictions
        
    def opt(self,corpus,c_values,sigma_values,r_values,s_values):
        # TODO: Change interface to take in the kernel type to be optimised as opposed to paramter values.
        # Those can be set in initialisation and read from e.g. self.c for linear kernel
        """
        This function optimises kernel parameters
        @param corpus: corpus on which the 10-fold cross-validation should be
        carried out
        @param c_values: values of the margin optimisation parameter 
        @param sigma_values: values of sigma for Gaussian kernel
        @param r_values,s_values: parameter values for quadratic kernel (s a \cdot b + r)^d
        """
        fold_accuracies = []
        std_devs = []
        if c_values and not sigma_values and not r_values and not s_values:
            for c in c_values:
                self.c = c
                self.crossValidate(corpus)
                fold_accuracy = self.pred_vec
                print ("10-fold accuracies for c= "+str(c)+"is:"+" "'[%s]' % ', '.join(map(str, fold_accuracy)))
                fold_accuracies.append(fold_accuracy)
                std_dev = self.getStdDeviation()
                print ("10-fold std deviation for c= "+str(c)+"is: "+str(std_dev))
                std_devs.append(std_dev)
            self.std_devs = std_devs
            self.accuracies = fold_accuracies
        if sigma_values:
            xx , yy = np. meshgrid (c_values,sigma_values)
            values_grid = np. concatenate (( xx. ravel (). reshape (( -1 , 1)) , \
    yy. ravel (). reshape (( -1 , 1))) , 1)
            for c, sigma in values_grid:
                self.c = c
                self.sigma = sigma
                print("C is: "+str(c))
                print("Sigma is: "+str(sigma))
                self.crossValidate(corpus)
                fold_accuracy = self.pred_vec
                print ("10-fold accuracies for c= "+str(c)+" and sigma= "+str(sigma)+"are:"+" "'[%s]' % ', '.join(map(str, fold_accuracy)))
                fold_accuracies.append(fold_accuracy)
                std_dev = self.getStdDeviation()
                print ("10-fold std deviation for c= "+str(c)+" and sigma= "+str(sigma)+"is: "+str(std_dev))
                std_devs.append(std_dev)
            self.std_devs = std_devs
            self.accuracies = fold_accuracies
        if r_values and c_values:
            xx , yy = np. meshgrid (r_values,s_values)
            values_grid = np. concatenate (( xx. ravel (). reshape (( -1 , 1)) , \
    yy. ravel (). reshape (( -1 , 1))) , 1)
            for c in c_values:
                for r, s in values_grid:
                    self.c = c
                    self.r = r
                    self.s = s
                    print("C is: "+str(c))
                    print("s is: "+str(s))
                    print("r is: "+str(r))
                    self.crossValidate(corpus)
                    fold_accuracy = self.pred_vec
                    print ("10-fold accuracies for c= "+str(c)+"s= "+str(s)+"and r= "+str(r)+"are:"+" "'[%s]' % ', '.join(map(str, fold_accuracy)))
                    fold_accuracies.append(fold_accuracy)
                    std_dev = self.getStdDeviation()
                    print ("10-fold std deviation for c= "+str(c)+"s= "+str(s)+"and r= "+str(r)+"is: "+str(std_dev))
                    std_devs.append(std_dev)
                self.std_devs = std_devs
                self.accuracies = fold_accuracies
class SVMText(SVM):
    def __init__(self,bigrams,trigrams,svmlight_dir,discard_closed_class,optimise):
        """ 
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        
        SVM.__init__(self,svmlight_dir,optimise)
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        self.optimise = optimise
    def getFeatures(self,reviews):
        """
        determine features from training reviews and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # reset for each training iteration
        self.vocabulary=set()
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review): 
                self.vocabulary.add(token)
        # using dictionary of vocabulary:index for constant order
        # features for SVMLight are stored as: (feature id, feature value)
        # using index+1 as a feature id cannot be 0 for SVMLight
        self.vocabulary={token:index+1 for index,token in enumerate(self.vocabulary)}

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def getVectors(self,reviews):
        """
        get vectors for svmlight from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of (string, list) tuples where string is the label ("1"/"-1") and list
                 contains the features in svmlight format e.g. ("1",[(1, 0.04), (2, 4.0), ...])
                 svmlight feature format is: (id, value) and id must be > 0.
        """
        
        #Set container for the data
        vectors = []
        #Set normalisation type
        normType = 1
        for sentiment, review in reviews:
            # Initialise containers for data processing at each step
            features = []
            counts = []
            # Check the label and convert it to string 
            if sentiment is "POS":
                label = 1
            elif sentiment is "NEG":
                label = -1
            # Count the number of occurences of the word in the review
            counts = collections.Counter(review)
            normalising_const = sum(counts.values())
            # Loop over each unique word in the review 
           # 2.7 for token in counts.iterkeys():
            for token in counts.keys():
                try:
                # Create a features structure which contains the word ID, taken from the vocabulary, along with the 
                # number of times the word occurs in that review (normalised to the length of the review)
                    # Compute mean counts
                    if normType is 1:
                        features.append(tuple([self.vocabulary[token],counts[token]*(100.0/normalising_const)]))
                        features.sort(key=lambda tup: tup[0])
                    elif normType is 2:
                        mean_counts = float(sum(counts.values()))/len(counts)
                        counts_sd = np.std(counts.values())
                        features.append(tuple([self.vocabulary[token],(counts[token]-mean_counts)/counts_sd]))
                        features.sort(key=lambda tup: tup[0])
                except:
                    continue
            # Append features and labels to the list containing all reviews  
           
            vectors.append(tuple([label,features]))
        return vectors
    
import math,sys

# All classifiers inherit from this class. This would be in the file Analysis.py
# Classifiers would be in Classfiers.py

class Evaluation():
    """ 
    general evaluation class implemented by classifiers 
    """
    def crossValidate(self,corpus):
        """
        function to perform 10-fold cross-validation for a classifier. 
        each classifier will be inheriting from the evaluation class so you will have access
        to the classifier's train and test functions. 

        1. read reviews from corpus.folds and store 9 folds in train_files and 1 in test_files 
        2. pass data to self.train and self.test e.g., self.train(train_files)
        3. repeat for another 9 runs making sure to test on a different fold each time

        @param corpus: corpus of movie reviews
        @type corpus: MovieReviewCorpus object
        """
        stack = []
        fold_pred = []
        for idx in range(10):
            train_files = []
            test_files = []
            train_fold_idx  = list(range(10))
            train_fold_idx.remove(idx)
            test_files = corpus.folds[idx]
            for fold_index in train_fold_idx:
                for review in corpus.folds[fold_index]:
                    train_files.append(review)
            self.train(train_files)
            self.test(test_files)
            for pred in self.predictions:
                stack.append(pred)
            test = self.getAccuracy()
            fold_pred.append(test)
        # reset predictions
        self.predictions = stack
        self.pred_vec = fold_pred
        # TODO Q3

    def getStdDeviation(self):
        """
        get standard deviation across folds in cross-validation.
        """
        # get the avg accuracy and initialize square deviations
        avgAccuracy,square_deviations=self.getAccuracy(),0
        # find the number of instances in each fold
        # Python 2.7: fold_size=len(self.predictions)/10
        fold_size=len(self.predictions)//10
        # calculate the sum of the square deviations from mean
        for fold in range(0,len(self.predictions),fold_size):
            square_deviations+=(self.predictions[fold:fold+fold_size].count("+")/float(fold_size) - avgAccuracy)**2
        # std deviation is the square root of the variance (mean of square deviations)
        return math.sqrt(square_deviations/10.0)

    def getAccuracy(self):
        """
        get accuracy of classifier. 

        @return: float containing percentage correct
        """
        # note: data set is balanced so just taking number of correctly classified over total
        # "+" = correctly classified and "-" = error
        return self.predictions.count("+")/float(len(self.predictions))


    