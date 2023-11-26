# # -----------------------------------------------------------------------------
# # Tissue Models Segmentation File
# # Author: Xavier Beltran Urbano and Fredderik Hartmann
# # Date Created: 09-11-2023
# # -----------------------------------------------------------------------------


import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os

from utils import Utils
class GaussianMixtureModel:
    def __init__(self, k, data, init_type, maxIterations=500, atlas=None, atlasInto=False):
        self.maxIterations = maxIterations
        self.data = data # (numberOfPixels, k)
        self.k = k # number of Clusters

        self.dimensions = data.shape[1]
        self.numberOfPixels = data.shape[0]

        self.membershipWeights = np.full((self.numberOfPixels, self.k), 1/self.k)
        self.mixtureWeights = np.full(self.k, 1/self.k)
        self.sumOfWeightedProbabilities = None

        self.covariance = np.array([self.randDiagCovarianceMatrix()for _ in range(self.k)])

        self.means = np.zeros((self.k, self.dimensions))

        self.currentLog = np.full(self.k, 1)
        self.prevLog = 0
        self.completedIterations = 0
        self.tolerance = 1e-6

        # division by zero
        self.epsilon = 10 * np.finfo(self.data.dtype).eps
        self.epsilonCovariance = np.zeros((self.dimensions, self.dimensions))
        np.fill_diagonal(self.epsilonCovariance, self.epsilon)

        # init
        self.atlas = atlas
        self.atlasInto = atlasInto
        self.initialization(init_type)

    def fit(self):
        # fits the GMM to the data
        while (not self.isConverged()):
            self.expectationStep()
            if self.atlasInto:
                self.membershipWeights*=self.atlas
            self.maximizationStep()
            self.completedIterations +=1
            print(f"Iteration {self.completedIterations} of {self.maxIterations}", end='\r')

    def predict(self):
        # return the predicted label
        return np.argmax(self.membershipWeights, axis=1)

    def predict_proba(self):
        # return the predicted probabilities
        return self.membershipWeights

    def initialization(self, initialization_type):
        # initialize the GMM with K-means, TissueModel, or TissueModel and Probabalistic Atlas
        if initialization_type == "KMeans":
            kmeans = KMeans(n_clusters=self.k, n_init="auto").fit(self.data)
            self.means = kmeans.cluster_centers_

        elif initialization_type == "TissueModel":
            # hard coded means
            self.means = [[43],[181], [104]]

        elif initialization_type == "Labelpropagation":
            # initialize means, covariances, mixture weights and memership weigths using the atlas
            if self.atlas is None: raise ValueError(f"Input the atlas as a numpy array")
            self.membershipWeights = self.atlas
            self.maximizationStep()

        elif initialization_type == "LabelpropagationAndTissueModel":
            # initialize means, covariances, mixture weights and memership weigths using the atlas
            if self.atlas is None: raise ValueError(f"Input the atlas as a numpy array")
            self.membershipWeights = self.atlas
            self.maximizationStep()
            self.means = [[43],[181], [104]]

        else:
            raise("allowed initialization types are Labelpropagation, TissueModel and KMeans")
        pass
    
    def expectationStep(self):
        # expectation step of the EM, updates membershipWeights
        weightedProbabilities = []
        for cluster in range(self.k):
            probability = self.gaussianDensityFunction(cluster)
            weightedProbability = probability * self.mixtureWeights[cluster]
            weightedProbabilities.append(weightedProbability)

        sumOfWeightedProbabilities = np.sum(weightedProbabilities, axis=0) 
        sumOfWeightedProbabilities = self.avoidDivisionByZero(sumOfWeightedProbabilities)

        for cluster in range(self.k):
            self.membershipWeights[:,cluster] = (weightedProbabilities[cluster] / sumOfWeightedProbabilities)[:,]
        self.currentLog = np.mean(np.log(sumOfWeightedProbabilities))

    def gaussianDensityFunction(self,cluster):
        # computes the gaussian density function using Cholesky decomposition
        det_cov = np.linalg.det(self.covariance[cluster])
        exponent = -0.5 * self.mahalanobisDistance(cluster)
        part1 = 1 / (((2*np.pi) ** (self.dimensions/2)) * (det_cov ** (1/2)))
        return part1 * np.exp(exponent)


    def mahalanobisDistance(self, cluster):
        # A faster implementation following the math of https://stats.stackexchange.com/a/147222
        L = np.linalg.cholesky(self.covariance[cluster])
        L_inv = np.linalg.inv(L)
        diff = self.data - self.means[cluster]
        y = L_inv @ diff.T
        mahaDist = np.square(y)
        return np.sum(mahaDist, axis=0)

    def maximizationStep(self):
        # expectation step of the EM, updates means, covariances, and mixture weights
        self.updateMeans()
        self.updateCovariance()
        self.updateMixtureWeights()

    def updateMeans(self):
        # updates means
        self.sumOfWeightedProbabilities = np.sum(self.membershipWeights, axis=0) + 10 * np.finfo(self.membershipWeights.dtype).eps
        self.means = np.dot(self.membershipWeights.T, self.data) / self.sumOfWeightedProbabilities[:, np.newaxis]
    
    def updateCovariance(self):
        # updates all covariance matrices
        for cluster in range(self.k):
            diff = self.data - self.means[cluster]
            self.covariance[cluster] = np.dot(self.membershipWeights[:,cluster] * diff.T, diff) / self.sumOfWeightedProbabilities[cluster]
            self.covariance[cluster] += self.epsilonCovariance

    def updateMixtureWeights(self):
        # updates mixture weights
        self.mixtureWeights = self.sumOfWeightedProbabilities / self.sumOfWeightedProbabilities.sum()

    def isConverged(self):
        # checks if the algorithm converged by comparing the new log with the previous log
        if self.completedIterations >= self.maxIterations:
            print(f"Maximum iterations ({self.maxIterations}) reached")
            return True
        elif (np.abs(self.currentLog - self.prevLog) < self.tolerance).all():
            print("\nconverged")
            return True
        else:
            self.prevLog = self.currentLog
            return False 

    def randDiagCovarianceMatrix(self):
        # return a random diagonal matrix
        np.random.seed(42)
        diagVector = np.random.uniform(5, 10, self.dimensions)
        return np.diag(diagVector)

    def avoidDivisionByZero(self, array):
        # checks if the array contains zeros and adds a small number if its true
        if array.any():
            array[array==0] += self.epsilon
        return array

    


if __name__ == "__main__":

    case = 1025
    testFolder = "images/testing-labels/"
    imageFolder = "images/testing-images/"

    utils=Utils()

    GT, affine = utils.readNiftiImage(testFolder + f"{case}_3C.nii.gz")
    T1,_ = utils.readNiftiImage(imageFolder + f"{case}.nii.gz")

    # Flatten and concatenate the data
    one_modality = T1[GT > 0].reshape(-1,1)
    normalizedImage = utils.normalizeImage(one_modality)

    n_classes=3
    # Start GMM Algorithm
    GMM = GaussianMixtureModel(k=n_classes, data=normalizedImage, init_type="KMeans")
    GMM.fit()
    cluster_assignments = GMM.predict()

    # Reconstruct Image
    reverted_image_ = utils.reconstruct_image(cluster_assignments, GT)
    reverted_image = utils.fitLabelToGT(reverted_image_.astype(np.uint8), GT.astype(np.uint8), n_classes)

    slice_index = 148  # Specify the slice index you want to display

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create a figure with two subplots

    # Plot reverted_image on the first subplot
    axes[0].imshow(reverted_image_[:, :, slice_index])
    axes[0].set_title('Reverted Image')

    # Plot reverted_image on the first subplot
    axes[1].imshow(reverted_image[:, :, slice_index])
    axes[1].set_title('matched Image')

    # Plot GT on the second subplot
    axes[2].imshow(GT[:, :, slice_index])
    axes[2].set_title('Ground Truth')

    plt.show()

    print(f"\n-----------------RESULTS------------------")
    print(f" CSF DSC: {utils.diceCoefficient((reverted_image == 1).astype(np.uint8), (GT == 1).astype(np.uint8))}")
    print(f" WM DSC: {utils.diceCoefficient((reverted_image == 2).astype(np.uint8), (GT == 2).astype(np.uint8))}")
    print(f" GM DSC: {utils.diceCoefficient((reverted_image == 3).astype(np.uint8), (GT == 3).astype(np.uint8))}")
