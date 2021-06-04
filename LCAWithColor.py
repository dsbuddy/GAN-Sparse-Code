import numpy as np
import matplotlib.pyplot as plt
import utils.plotFunctions as pf
import utils.helperFunctions as hf
from sklearn.decomposition import SparseCoder

imageSize = (32,32,3)
dictSize = 64
sampleSize = 100




# Get dictionary from CIFAR-10 file
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# dictionary with vectorized RGB images and labels, (images as vectors of pixel values, [r, g, b])
dict = unpickle("cifar-10-batches-py/test_batch")
images = dict[b'data'][0:sampleSize]

def patches2FullImage(patches):
    red = []
    green = []
    blue = []
    for patch_id in range(len(patches[0])): # for each patch
        patch = patches[:,patch_id]
        red_layer = np.reshape(patch[0:64],(8,8))
        green_layer = np.reshape(patch[64:128],(8,8))
        blue_layer = np.reshape(patch[128:192],(8,8))
        if len(red)==0:
            red = red_layer
            green = green_layer
            blue = blue_layer
        else:
            red = np.concatenate((red,red_layer),axis=1)
            green = np.concatenate((green,green_layer),axis=1)
            blue = np.concatenate((blue,blue_layer),axis=1)
    i = 0
    offset = i*32
    red_layer = np.concatenate((red[:,0+offset:32+offset],red[:,32+offset:64+offset],red[:,64+offset:96+offset],red[:,96+offset:128+offset]),axis=0)
    green_layer = np.concatenate((green[:,0 + offset:32], green[:,32 + offset:64], green[:,64 + offset:96], green[:,96 + offset:128]),axis=0)
    blue_layer = np.concatenate((blue[:,0 + offset:32], blue[:,32 + offset:64], blue[:,64 + offset:96], blue[:,96 + offset:128]),axis=0)

    red = []
    green = []
    blue = []
    for i in range(len(red_layer)):
        red = np.concatenate((red,red_layer[i,:]))
        green = np.concatenate((green, green_layer[i, :]))
        blue = np.concatenate((blue, blue_layer[i, :]))

    reconstruction = np.concatenate((red,green,blue))
    return reconstruction


# Sparse Coding hyperparameters
numTrials = 1  # Number of weight update steps
numInputs = 192  # pixels
numOutputs = 64  # number of neurons (dictionary size)
lambdav = 0.5   # activation threshold
batchSize = 16  # Number of samples to use in a single weight update
eta = 0.1 # Learning rate

# Inference hyperparameters
tau = 100 # LCA update time constant
numInferenceSteps = 400 # Number of iterations to run LCA

# Plot display parameters
displayInterval = 400 # How often to update display plots during learning
displayStatsInterval = 10  # How often to update the stats plots
#
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
    return a


def inference(dataBatch, phi, tau, lambdav, numInferenceSteps, u_prev,trial,image_num,mean=0,std=1,):
    """
    Compute

    Parameters
    ----------
    dataBatch : Batch of data samples, shape=(numInputs, batchSize)
    phi : Dictionary, shape=(numInputs, numOutputs)
    tau : Time Constant of LCA update, scalar float
    lambdav : Both the sparsity tradeoff term and the LCA Threshold value
    numInferenceSteps: Number of inference steps to take

    Returns
    -------
    a : Activity, i.e. thresholded potentials, shape=(numOutputs, batchSize)
    """
    b = phi.T @ dataBatch # Driving input (64 x 16)
    gramian = phi.T @ phi - np.identity(int(phi.shape[1])) # Explaining away matrix (64 x 64)
    u = u_prev
    # u = np.zeros_like(b) # Initialize membrane potentials to 0

    a = threshold(u, lambdav) # Activity vector contains thresholded membrane potentials
    du = b - u - gramian @ a # LCA dynamics define membrane update (equation (5))
    u += (1.0 / tau) * du # Update membrane potentials using time constant
    # if  (trial == 0) and (image_num == 1):
    #     reconstruction = patches2FullImage(phi @ a)
    #     imgMat = pf.vec2RGB(reconstruction * std + mean,-1)
    #     plt.imshow(imgMat)
    #     plt.axis('off')
    #     plt.savefig("frames/%d/"%image_num + str(trial).zfill(3) + ".png", dpi=300)
    return u, threshold(u, lambdav)


def weightUpdate(phi, dataBatch, activity, learningRate):
    dataBatchReconstruction = phi @ activity # Reconstruct input using the neuron activity (stimulus approximation)
    reconstructionError =  dataBatch - dataBatchReconstruction # Error between the input and reconstruction
    dPhi = reconstructionError @ activity.T  # Weight update rule (dE/dPhi)

    phi = phi + learningRate * dPhi # Scale weight update by learning rate
    return (phi, reconstructionError)


def fullImage2Patches(image):
    # Make batch of random image
    patches = np.zeros((numInputs, batchSize))  # getting sequential 8x8 RGB patches (192 pixels each)
    red = np.reshape(image[0:1024],(32,32))
    green = np.reshape(image[1024:2048],(32,32))
    blue = np.reshape(image[2048:3072],(32,32))

    for i in range(4):
        for j in range(4):
            red_layer = red[0 +8 *i:8 + 8*i,0 + 8*j:8 + 8*j].flatten()
            green_layer = green[0 + 8*i:8 + 8*i, 0 + 8*j:8 + 8*j].flatten()
            blue_layer = blue[0 + 8*i:8 + 8*i, 0 + 8*j:8 + 8*j].flatten()
            patches[:,i*4+j] = np.concatenate((red_layer,green_layer,blue_layer))


    return patches

def trainModel(dataset):
    # # Plotting
    MSE = []
    sparsityPercentage = []
    # Initialize Phi weight matrix with random values
    np.random.seed(0)
    phi = hf.l2Norm(np.random.randn(numInputs, numOutputs))

    prevFig = pf.plotDataTiled(phi, "Dictionary at image 0", None)
    learningRate = eta / batchSize
    activity_prev = np.zeros((64,16))
    # Do sparse coding with LCA
    # for image_num in range(len(dataset)):
    for image_num in range(sampleSize):
        image = dataset[image_num]
        std = np.std(image)
        mean = np.mean(image)
        image = (image - mean) / std

        # Make batch of random image
        dataBatch = fullImage2Patches(image)
        if image_num == 0:
            u_prev = np.zeros((64,16))
        for trial in range(numTrials):
            # Compute sparse code for batch of data samples

            if trial == (numTrials-1):
                u_prev, activity = inference(dataBatch, phi, tau, lambdav, numInferenceSteps,u_prev,trial,image_num,mean,std)
            else:
                _, activity = inference(dataBatch, phi, tau, lambdav, numInferenceSteps,u_prev,trial,image_num,mean,std)

        sparsityPercentage.append((64*16-np.count_nonzero(activity))/(64*16))
        # Update weights using inferred  sparse activity
        (phi, reconstructionError) = weightUpdate(phi, dataBatch, activity, learningRate)
        # Renormalize phi matrix
        phi = hf.l2Norm(phi)

        X_std = (phi-phi.min(axis=0)) / (phi.max(axis=0) - phi.min(axis=0))
        X_scaled = X_std * 255

        if image_num == 0 or image_num == 9 or image_num == 49 or image_num == 99:
            prevFig = pf.plotDataTiled(X_scaled, "Dictionary at image " + str(image_num), prevFig)
            # prevFig[0].savefig('dictionaryAt' + str(image_num+1))
            prevFig[0].show()
        reconstruction = patches2FullImage(phi @ activity)
        reconstructionMSE = sum(((reconstruction - image)**2))/3072
        MSE.append(reconstructionMSE)
        print(reconstructionMSE)
        if image_num == 0 or image_num == 9 or image_num == 49 or image_num == 99:
            imgMat = pf.vec2RGB(reconstruction*std+mean,-1)
            plt.imshow(imgMat)
            plt.show()
            plt.savefig('reconstruction'+str(image_num+1))

    # x = range(sampleSize)
    # plt.plot(x,MSE)
    # plt.xlabel('Image #')
    # plt.xlim((0,sampleSize-1))
    # plt.xticks(np.arange(0,sampleSize,10))
    # plt.ylabel('MSE')
    # plt.ylim((0,1))
    # plt.yticks(np.arange(0,1,0.1))
    # plt.title('Reconstruction MSE Over Time')
    # # plt.savefig('MSEVsTime')
    # plt.show()
    # plt.plot(x,sparsityPercentage)
    # plt.xlabel('Image #')
    # plt.xlim((0, sampleSize - 1))
    # plt.xticks(np.arange(0, sampleSize, 10))
    # plt.ylabel('Sparsity (Percentage)')
    # plt.ylim((0, 1))
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.title('Weight Sparsity Over Time')
    # plt.show()
    # plt.savefig('SparsityVsTime')
    # file = open("dictionary","wb")
    # np.save(file,phi)
    # print(np.shape(phi))
    # file.close()
trainModel(images) # CIFAR-10

def test_plot():
    file = open("dictionary","rb")
    dict = np.load(file)
    file.close()
    print(np.shape(dict))
    coder = SparseCoder(dict.T)
    print(np.shape(images))

    codes = []
    for i in range(sampleSize):
        img = fullImage2Patches(np.array(images[i]))
        code = coder.transform(img.T)
        codes.append(code)
    # rgb = pf.vec2RGB(images[0])
    reconstruction = code @ dict.T
    rgb = pf.vec2RGB(patches2FullImage(reconstruction.T))
    file = open('sparse_codes','wb')
    np.save(file,codes)
    file.close()
    plt.imshow(rgb)
    plt.show()
test_plot()
