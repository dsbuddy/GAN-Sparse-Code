import numpy as np
from time import time
import math
import utils.plotFunctions as pf
import utils.helperFunctions as hf

#Functions from Berkley, Neural Computation, lab 4
def threshold(u, lambdav):                                                                                                                                                                                                          
    """   
    Compute the activity of the neurons using the membrane potentials (u) using soft thresholding:                                                                                                                                           
                                                                                                                                                                                                                                         
    a = T(u) = u - threshold, u > threshold                                                                                                                                                                                              
               u + threshold, u < -threshold                                                                                                                                                                                             
               0, otherwise                                                                                                                                                                                                              
    """
    a = np.abs(u) - lambdav                                                                                                                                                                                                            
    a[np.where(a<0)] = 0          
    a = np.sign(u) * a    
    a[np.where(a<0)] = 0 #Non-negative only
    return a

def inference(dataBatch, phi, tau, lambdav, numInferenceSteps):
    """
    Compute

    Parameters
    ----------
    dataBatch : Batch of data samples, shape=(numInputs, batchSize) 
    phi : Dictionary, shape=(numInputs, numOutputs)
    tau : Time Constant of LCA update, scalar float
    lambdav : Both the sparsity tradeoff term and the LCA Threshold value
    numInferencSteps: Number of inference steps to take
    
    Returns
    -------
    a : Activity, i.e. thresholded potentials, shape=(numOutputs, batchSize)
    """
    u_history = [] # List of membrane potentials recorded at each integer 2^i
    
    b = phi.T @ dataBatch # Driving input
    gramian = phi.T @ phi - np.identity(int(phi.shape[1])) # Explaining away matrix
    u = np.zeros_like(b) # Initialize membrane potentials to 0
    
    for step in range(numInferenceSteps):
        a = threshold(u, lambdav) # Activity vector contains thresholded membrane potentials
        du = b - u - (gramian @ a)
        u += (1.0 / tau) * du # Update membrane potentials using time constant
        
        # If step is a power of 2 (2, 4, 6... 256... numInferenceSteps), record membrane potentials
        if step != 0 and (math.ceil(np.log2(step)) == math.floor(np.log2(step))) or step == (numInferenceSteps - 1): 
            #print("Recording membrane potentials at step: ", step)
            u_history.append(u)
            
    return u_history, threshold(u, lambdav)

def weightUpdate(phi, dataBatch, activity, learningRate):
    dataBatchReconstruction = phi @ activity
    reconstructionError = dataBatch - dataBatchReconstruction
    dPhi = reconstructionError @ activity.T # Weight update rule (dE/dPhi)
    phi = phi + learningRate * dPhi # Scale weight update by learning rate
    return (phi, reconstructionError)


# Computes dictionary with LCA
# Function from Berkley, Neural Computation, lab 4
def trainModel(dataset, lambdav, batchSize, eta, numTrials, tau, numInferenceSteps, displayInterval, displayStatsInterval, numInputs, numOutputs, numDataPoints):
    t0 = time()
    # Plotting
    sumPercentNonZero, sumEnergy, sumReconstructionQualitySNR = 0, 0, 0
    statsFig, statsAxes = pf.plotStats(numTrials)
    
    # Initialize phi weight matrix randomly
    phi = hf.l2Norm((np.random.randn(numInputs, numOutputs) * 2) - 1)
    
    # Sparse coding with LCA
    for trial in range(numTrials):
        # Make batch of random images
        dataBatch = np.zeros((numInputs, batchSize))
        for batchNum in range(batchSize):
            dataBatch[:, batchNum] = dataset[np.random.randint(dataset.shape[0])]
            
        # Compute sparse code for batch of data samples
        _, activity = inference(dataBatch, phi, tau, lambdav, numInferenceSteps)

        # Update weights using inferred sparse activity
        learningRate = eta / batchSize
        (phi, reconstructionError) = weightUpdate(phi, dataBatch, activity, learningRate)

        # Renormalize phi matrix
        phi = hf.l2Norm(phi)

        # Record stats for plotting
        percentNonZero, energy, reconstructionQualitySNR = (
            hf.computePlotStats(activity, reconstructionError, lambdav))
        sumPercentNonZero += percentNonZero
        sumEnergy += energy
        sumReconstructionQualitySNR += reconstructionQualitySNR

        if trial and trial % displayStatsInterval == 0:
            avgEnergy, avgPercentNonZero, avgReconstructionQualitySNR = (
                sumEnergy/ displayStatsInterval, sumPercentNonZero / displayStatsInterval, 
                sumReconstructionQualitySNR / displayStatsInterval)
            pf.updateStats(statsFig, statsAxes, trial, avgEnergy, avgPercentNonZero, avgReconstructionQualitySNR)
            sumPercentNonZero = 0
            sumEnergy = 0
            sumReconstructionQualitySNR = 0
    dt = time() - t0    
    print("Dictionary learned in: ", dt)
            
    return phi