# -*- coding: utf-8 -*-
"""
Created on Mon May 14 20:03:17 2018

@author: alexc
"""



def splitSet (y,trainFraction):
    '''Function that splits a set into training/testing sets
    Input: y, training labels
    trainFraction: % of examples used in training
    Returns:     
    
    trainingSet (binary vector): a value of 1 indicates that the example 
    with the corresponding index is part of the training set
    
    testingSet (binary vector): a value of 1 indicates that the example with 
    the corresponding index is part of the testing set
    '''
    # Set random seed
    np.random.seed(seed=2)
    # Choose random integers from the [0,len(y)] interval. Choose only once so replacement not needed
    trainingSet = np.random.choice(np.arange(0,len(y),1),int(trainFraction*len(y)),replace=False)
    # Create boolean mask for the training set - used to select the test set
    mask = np.zeros(np.arange(0,len(y),1).shape,dtype=bool)
    mask[trainingSet] = True
    # Select the testing set
    testSet =np.arange(0,len(y),1)[~mask]
    return (trainingSet,testSet)


# Train regularised logistic regression

def betaMAP(X,y,alpha,beta,maxIter,tol,opt):
    ''' Regularised logistic regression training
    Inputs:
        alpha: learning rate
        beta : initial paramter values 
        X,y: training data (X = features, y=labels)
        maxIter: maximum iterations 
        tol: tolerance (for stopping condition)
        opt: if getParamIter, returns all parameters intermendiate vals 
         (used to plot e.g., log-likelihood)
    Outputs:
        beta: paramters
        betaMat: values of parametrs 
    '''
    count = 0
    betaMat= np.zeros((len(beta),maxIter+1))
#    grad = np.matmul(np.transpose(X),(y-sigmoid(X,beta)))
    while True:
         # (Vectorised) Gradient implementation 
         grad = np.sum((y-sigmoid(X,beta))[:,np.newaxis]*X,0)
         # Parameter update (simple: alpha is fixed)
         beta = beta*(1-alpha) + grad*alpha
#         if count % 2 == 0:
         betaMat[:,count] = beta
         checkpoints = np.arange(0,maxIter+10000,1000)
         if count in checkpoints:
             print((np.linalg.norm(grad,2),likelihood(X,y,beta,0,0),count))
         # Check stopping condition (based on the relative error of the parameter vector )
         if ((np.linalg.norm(betaMat[:,count]-betaMat[:,count-1],2)/np.linalg.norm(betaMat[:,count],2) < tol) and count>1) or (count >= maxIter):
             break
         count+=1
    if opt == 'getParamIter':    
        return (beta,betaMat)
    else:   
        return (beta,0)
    
def augmentX (X):
    ''' Add a column of 1s to the training set, X (to account for biases)''''
    newX = np.ones((np.shape(X)[0],np.shape(X)[1]+1))
    newX[:,1:np.shape(X)[1]+1] = X
    return newX

def removeZeroCols (X,tol):
    '''Removes columns with sum = 0 from matrix X'''
    mask = np.zeros((1,np.shape(X)[1]),dtype=bool)
    for i in range(np.shape(X)[1]):
        if np.linalg.norm(X[:,i],2)<=tol:
            mask[0,i]=True
        else:
            mask[0,i]=False
    return X.compress(~mask[0],axis=1)

def likelihood(X,y,beta,betaMat,opt):
    ''' Implements log-likelihood for logistic regression
    Input: 
        X,y: training data
        beta,betaMat: current paramters/parameters evolution
        opt: used to return log-likelihood curve '''
    #Numerically stable
    # Helper function: vectorised implementation of sigmoid
    def sigmoid_helper(X):
        func = np.vectorize(lambda x: 1/(1+np.exp(-x)))
        return func(X)
    y1 = np.ones(len(y))-y
    ll = 0
    # likelihoodEv is used to plot the log-likelihood
    if opt == 'retLikelihoodEv':
        likelihoodEv = np.multiply(1/len(y),np.sum(np.log(sigmoid_helper(np.matmul(X,betaMat)))*y[:,np.newaxis] -y1[:,np.newaxis]*np.matmul(X,betaMat)+y1[:,np.newaxis]*np.log(sigmoid_helper(np.matmul(X,betaMat))),0))
        ll = np.multiply(1/len(y),np.sum(y*np.log(sigmoid(X,beta))) -\
             np.sum((np.ones(len(y))-y)[:,np.newaxis]*beta*X)+\
             np.sum((np.ones(len(y))-y)*np.log(sigmoid(X,beta))))
        return (ll,likelihoodEv)
    else:
#        (np.sum(y*np.log(sigmoid(X,beta))) + np.sum((np.ones(len(y))-y)*(np.log(np.ones(len(y))-sigmoid(X,beta)))),0)
         # Numerically stable implementation 
         ll = np.multiply(1/len(y),np.sum(y*np.log(sigmoid(X,beta)))-\
               np.sum((np.ones(len(y))-y)[:,np.newaxis]*beta*X)+\
               np.sum((np.ones(len(y))-y)*np.log(sigmoid(X,beta)),0))
         return ll


def predictLabel(X,beta,threshold=0.5):
    ''' Predict labels for the data stored in matrix X given parameters in beta 
    and Threshold = threshold''' 
    prediction = np.ndarray(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
#        pdb.set_trace()
        if sigmoid(X,beta)[i]>threshold:
            prediction[i]=1
        else:
            prediction[i]=0
    return prediction

def confMatrix(y,prediction):
    ''' Calculate confusion matrix given the true labels y and the predictions 
    in prediction
    Output: tuple with (true positives, true negatives, false positives, false negatives)
    NB: These should have been normalised...and should have been returned as a list to avoid hassle in ROC function''' 
    tp,fn,fp,tn = 0,0,0,0
    for entry in zip(y.astype(int),prediction):
        if entry[0] == entry[1] and (entry[0]) == 1:
            tp += 1
        elif entry[0] > entry[1]:
            fn += 1
        elif entry[0] < entry[1]:
            fp += 1
        else:
            tn += 1
    return (tp,tn,fp,fn)

def calculateROC (X,y,beta,threshold_grid,tol):
    ''' Calculates the ROC curve for the logistic regression classifier
    Inputs:
        X,y: training data
        threshold_grid: range of thresholds for which the ROC is to be calculated
        tol: Used to establish if two consecutive values are different or not
    Outputs:
        ROC and and corresponding area under curve (AUC)'''
    ROC = np.ndarray((2,len(threshold_grid)))
    cont = np.ndarray((1,np.shape(ROC)[1]-1))
    index = 0
    for threshold in threshold_grid:
        # Make predictions
        prediction = predictLabel(X,beta,threshold)
        # Construct confusion matrix
        confmatrix = confMatrix(y,prediction)
        # Cumbersome because confmatrix returns a tuple...
        confmatrix = tuple([0.01*x for x in list(confmatrix)])
        ROC[:,index] = np.array([confmatrix[0],confmatrix[2]])
        index +=1
    for i in np.arange(np.shape(ROC)[1]-1):
        if abs(ROC[0,i]-ROC[0,i+1]) < tol :
            cont [0,i] = ROC[0,i]*abs((ROC[1,i+1]-ROC[1,i]))
        else:   
            cont [0,i] = 0.5*(ROC[0,i]+ROC[0,i+1])*abs((ROC[1,i+1]-ROC[1,i]))
    AUC = np.sum(cont)
    return (ROC,AUC)

# Train logistic regression using Python inbuilt tools

#Define objective and Jacobian of the objective
grad1 = lambda beta,variance,X,y: - np.sum((y-sigmoid(X,beta))[:,np.newaxis]*X,0) + np.multiply(beta,1/variance)
func1 = lambda beta,variance,X,y: - (np.sum(y*np.log(sigmoid(X,beta))) - np.sum((np.ones(len(y))-y)[:,np.newaxis]*beta*X)+np.sum((np.ones(len(y))-y)*np.log(sigmoid(X,beta)),0)-(0.5/variance)*np.sum(beta*beta)-(0.5)*(len(y)+1)*np.log(np.pi)-(len(y)+1)*0.5*np.log(variance))
# Run L-BFGS optimisation for MAP estimate
def calculateMAP (X,y,initial_param,variance):    
    res = optimizers.minimize(func1,args=(variance,X,y),x0=initial_param,method='BFGS',jac=grad1,options={'gtol': 1e-5, 'disp': True})
    betaMap = res.x
    return betaMap

#
def expand_inputs (l, X, Z):
    '''Performs radial basis function expansion of the data points in X at points in Z.
    Here l is the lengthscale of the expansion
    Output: r2 is a matrix containing the transformed data points (size is (X.shape[0],Z.shape[0])'''
    X2 = np. sum (X**2 , 1)
    Z2 = np. sum (Z**2 , 1)
    ones_Z = np. ones (Z. shape [ 0 ])
    ones_X = np. ones (X. shape [ 0 ])
#    pdb.set_trace()
    r2 = np. outer (X2 , ones_Z ) - 2 * np. dot (X, Z.T) + np. outer ( ones_X , Z2)
    return np. exp ( -0.5 / l **2 * r2)

# Bayesian logistic regression
# See Murphy's book (pag. 257-263 for the equations implemented here)
def compQF (X,H):
'''This function computes the inverse of the Hessian matrix,H in a numerically stable way.
Advantage is taken of the fact that the product XHinvX' has to be calculated.'''
    # Numerically stable computation of XHinvX'
    # Compute the Cholesky decomposition of the Hessian
    L = linalgebra.cholesky(H, lower=True)
    # Solve the linear system L*temp = X to calculate L^-1@X
    temp = linalgebra.solve_triangular(L,np.transpose(X))
    # Solve the linear sytem L'temp1 = temp to calculate L^-T*L^-1*X
    temp1 = linalgebra.solve_triangular(np.transpose(L),temp)
    #Premultiply by X to get the right result
    res = X @ temp1
    return res

def hess (X,beta,variance):
    '''Hessian Matrix for logistic regression. Equation 4.97 (p. 207) in Bishop but regularised'''
    t = sigmoid(X,beta)
    t1 =1 - sigmoid(X,beta)
    D = np.diag(t*t1)
    hess = np.matmul(np.matmul(np.transpose(X),D),X) + (1/variance)*np.identity(beta.shape[0])
    return hess


def makeBayesianPred(X,y,beta,variance,length_scale):
    ''' This function predicts the labels of the points in X using Bayesian Logistic Regression.
    This is a vectorised implementation of equations (8.66),(8.64),(8.70),(8.71) (pag. 262-263) in Murphy's book.
    Output: a vector containing p(y=1|x,D) using an approximation to the posterior predictive distribution'''
    # Compute hessian matrix
    H = hess(X,beta,variance)
    vec1 = compQF(X,H)
    vec2 = np.diagonal(vec1)
    # Eq (8.70)
    vec3 = 1/np.sqrt(np.multiply(1/8,(vec2*np.pi))+np.ones(vec2.shape))
    # Eq (8.64)
    vec4 = np.sum(beta[:,np.newaxis]*np.transpose(X),0)
    vec5 = vec3*vec4
    out = 1/(1+np.exp(-vec5)) # Eq (8.71)
    return out

def modelEvidence(X,y,initial_param,variance,length_scale,no_features):
    '''This function calculates the Laplace approximation to the log marginal likelihood. Eq (8.55) in Murphy (pag. 258)
    Inputs: 
        initia_param: initial value of model paramters, to initialise MAP parameter search
        variance: variance of the prior over the model parameters
        length_scale: length scale of the basis function expansion
        no_features: number of features - Xtraining.shape[0]+1'''
    # Perform basis epansion at the training data points    
    X = augmentX(expand_inputs(length_scale,X,Xtraining[:,1:3]))
    # Calculate MAP estimate of the model paramters
    beta = calculateMAP(X,y,initial_param,variance)
    # Calculate Hessian matrix
    H = hess(X,beta,variance)
    L = linalgebra.cholesky(H, lower=True)
    # Calculate the log of the determinant of the Hessian matrix
    detHess = np.prod(np.diag(L)**2)
    # Compute the model evidence
    evidence = -func1(beta,variance,X,y) + no_features*0.5*np.log(2*np.pi)-0.5*detHess
    return evidence

# Run grid search to optimise RBF expansion and prior variance
    
def createGrid (variances,length_scales):
    ''' This function creates a grid of values for the hyperparamters variance and length scales
    from the vectors specified in the input'''
    xx, yy= np.meshgrid(variances,length_scales)
    # Convert grid into an array of points
    points = np. concatenate (( xx. ravel (). reshape (( -1 , 1)) , \
        yy. ravel (). reshape (( -1 , 1))) , 1)
    return points

def runGridSearch(points,Xinit,no_features):
    '''This function performs a hyper-paramter search using the points specified in the vector points 
    (NB: these are obtained with createGrid)
    Output:
        full_data: A vector containing the log marginal-likelihood for the data. The maximum log marginal likelihood
        is used as a selection criterion for the optimal hyperparameters. The hyper parameter combination is also stored along 
        with the value of the log marg. likelihood''' 
    evidenceVec = np.zeros(points.shape[0])
    full_data = []
    for i in range(points.shape[0]):
        X = Xinit
        evidenceVec[i] = modelEvidence(X,y,beta_initial,points[i,0],points[i,1],no_features)
        full_data.append((evidenceVec[i],points[i,0],points[i,1]))
        print('Evidence for model with length_scale'+' '+str(points[i,1])+' '+'and variance'+' '+str(points[i,0])+'is'+\
              ' '+str(evidenceVec[i]))
    return full_data