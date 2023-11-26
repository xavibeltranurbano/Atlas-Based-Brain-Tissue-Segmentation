# # -----------------------------------------------------------------------------
# # Tissue Models Segmentation File
# # Author: Xavier Beltran Urbano and Fredderik Hartmann
# # Date Created: 09-11-2023
# # -----------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import gc
import numpy as np
import pandas as pd
import time
import nibabel as nib

from GMM import GaussianMixtureModel
from utils import Utils

class InitializationTest:
    utils = Utils()

    def __init__(self):
        self.imageFolder = "images/testing-images"
        self.labelFolder = "images/testing-labels"
        self.atlasFolder = "registeredAtlases"

        self.imagePaths= self.utils.getAllFiles(self.imageFolder)
        self.labelPaths = self.utils.getAllFiles(self.labelFolder)
        self.atlasPaths = self.utils.getAllFiles(self.atlasFolder)

        self.nClasses = 3

        # metric storage
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}



    ############################################
    ### TESTS ##################################
    ############################################

    def kmeansInitTest(self):
        # GMM with kmeans initialization    
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}    
        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="KMeans")

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterAssignments = gmm.predict()
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/KmeansInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("initKmeans.csv")

    def tissueModelInitTest(self):
        # GMM with kmeans initialization
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}        
        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="TissueModel")

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterAssignments = gmm.predict()
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/tissueModelInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("initTissueModel.csv")

    def labelPropInitTest(self):
        # GMM with atlas initialization
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}        
        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            atlas = self.loadAtlasCorrespondingTo(currentImagePath)
            maskedAtlas = self.maskAtlas(atlas, labelImage)

            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="Labelpropagation", atlas=maskedAtlas)

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterAssignments = gmm.predict()
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/labelPropInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("initLabelpropagation.csv")

    def atlasAfterBestInitTest(self):
        # GMM with tissue models initialization and atlas after
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}

        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            atlas = self.loadAtlasCorrespondingTo(currentImagePath)
            maskedAtlas = self.maskAtlas(atlas, labelImage)

            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="TissueModel", atlas=maskedAtlas)

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterProb = gmm.predict_proba()
            clusterProb *= maskedAtlas
            clusterAssignments = np.argmax(clusterProb, axis=1)
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/atlasAfterBestInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("atlasAfterBestInit.csv")
    
    def atlasAfterLabelAndTissueInitTest(self):
        # GMM with Labelpropagation and TissueModel initialization and atlas after
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}

        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            atlas = self.loadAtlasCorrespondingTo(currentImagePath)
            maskedAtlas = self.maskAtlas(atlas, labelImage)

            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="LabelpropagationAndTissueModel", atlas=maskedAtlas)

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterProb = gmm.predict_proba()
            clusterProb *= maskedAtlas
            clusterAssignments = np.argmax(clusterProb, axis=1)
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/atlasAfterLabelAndTissueInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("atlasAfterLabelAndTissueInit.csv")

    def atlasIntoBestInitTest(self):
        # GMM with tissue models initialization and atlas integrated at every iteration
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}

        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            atlas = self.loadAtlasCorrespondingTo(currentImagePath)
            maskedAtlas = self.maskAtlas(atlas, labelImage)

            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="TissueModel", atlas=maskedAtlas, atlasInto=True)

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterAssignments = gmm.predict()
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/atlasIntoBestInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("atlasIntoBestInit.csv")

    def atlasIntoLabelAndTissueInitTest(self):
        # GMM with Labelpropagation and TissueModel initialization and atlas integrated at every iteration
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time' : []}

        for i, currentImagePath in enumerate(self.imagePaths):
            print(f"_"*40)
            print(f"segmenting image {os.path.basename(currentImagePath)}")
            labelPath = self.matchImagePathToLabelPath(currentImagePath)
            normImageNoBackground, labelImage, affine = self.initDataForGMM(labelPath, currentImagePath)
            atlas = self.loadAtlasCorrespondingTo(currentImagePath)
            maskedAtlas = self.maskAtlas(atlas, labelImage)

            gmm = GaussianMixtureModel(k=self.nClasses, data=normImageNoBackground, init_type="LabelpropagationAndTissueModel", atlas=maskedAtlas, atlasInto=True)

            # fit gmm
            startTime = time.time()
            gmm.fit()
            clusterAssignments = gmm.predict()
            endTime = time.time()

            # metrics
            segmentedImage = self.rebuildImage(clusterAssignments, labelImage)
            self.computeMetrics(segmentedImage, labelImage, endTime-startTime)

            # Save Image
            folderPath = "Results/atlasIntoLabelAndTissueInit"
            self.saveImage(segmentedImage, affine, folderPath, currentImagePath)

            # reset loop
            del gmm
            gc.collect()

        self.saveMetrics("atlasIntoLabelAndTissueInit.csv")
        


    ############################################
    ### METRICS ################################
    ############################################

    def computeMetrics(self, segmentedImage, gt, timeDiff):
        self.metrics['WM'].append(self.utils.diceCoefficient(segmentedImage == 1, gt == 1))
        self.metrics['CSF'].append(self.utils.diceCoefficient(segmentedImage == 2, gt == 2))
        self.metrics['GM'].append(self.utils.diceCoefficient(segmentedImage == 3, gt == 3))
        self.metrics['Time'].append(timeDiff)



    ############################################
    ### HELPER FUNCTIONS #######################
    ############################################

    def initDataForGMM(self, labelPath, currentImagePath):
        # data transformation (masking, normalization, reshaping)
        labelImage, _ = self.utils.readNiftiImage(labelPath)
        currentImage, affine = self.utils.readNiftiImage(currentImagePath)
        normImageNoBackground = self.preprocess(currentImage, labelImage)
        return normImageNoBackground, labelImage, affine
    
    def preprocess(self, image, labels):
        # removes background, normalizes and returns a vector
        maskedImage = image[labels > 0]
        normImage = self.utils.normalizeImage(maskedImage)
        vecImage = normImage.reshape(-1,1)
        return vecImage


    def matchImagePathToLabelPath(self, imagePath):
        # matches the imagePath to the corresponding label (GT) and return the labelPath
        imageName = os.path.basename(imagePath).split(".")[0]
        for labelPath in self.labelPaths:
            labelName = os.path.basename(labelPath).split("_")[0]
            if labelName in imageName:
                return labelPath
        raise FileNotFoundError(f"No matching label file for: {imagePath} in {self.labelFolder} found.")

    def rebuildImage(self, clusterAssignments, labelImage):
        # rebuilds the image from the vector of clusterAssignments
        reconstructedImage = self.utils.reconstruct_image(clusterAssignments, labelImage)
        switchedLabels = self.utils.fitLabelToGT(reconstructedImage.astype(np.uint8), labelImage.astype(np.uint8), self.nClasses)
        return switchedLabels

    def loadAtlasCorrespondingTo(self, imagePath):
        # returns the registered atlas (numpy array) to the corresponding imagePath
        imageName = os.path.basename(imagePath).split(".")[0]
        for atlasPath in self.atlasPaths:
            atlasName = os.path.basename(atlasPath).split("_")[2].split(".")[0]
            if atlasName in imageName:
                atlas, _ = self.utils.readNiftiImage(atlasPath)
                return atlas
        raise FileNotFoundError(f"No matching atlas file for: {imagePath} in {self.atlasFolder} found.")

    def maskAtlas(self, atlas, labelImage):
        # masks the atlas
        labels = atlas.shape[3]
        numberOfPixels = len(atlas[..., 0][labelImage > 0])

        atlases = np.zeros((numberOfPixels, labels))
        for tissue in range(labels):
            oneTissueAtlas = atlas[..., tissue]
            atlases[:,tissue] = oneTissueAtlas[labelImage > 0]

        return atlases



    ################################################
    ################ Save Results ##################
    ################################################
    def saveMetrics(self, fileName):
        # saves all metrics in csv file
        dice_df = pd.DataFrame(self.metrics)
        mean_values = dice_df.mean()  # Last row contains the mean of each column
        dice_df = dice_df._append(mean_values, ignore_index=True)
        dice_df.to_csv('Results/' + fileName, index=False)

    def saveImage(self, image, affine, folderPath, imagePath):
        # safes image and metata in a nii.gz file
        newImage = nib.Nifti1Image(image, affine)
        imageName = os.path.basename(imagePath).split(".")[0]
        imageName = imageName + "_3C.nii.gz"

        self.utils.ensureFolderExists(folderPath)
        storePath = os.path.join(folderPath, imageName)
        newImage.to_filename(storePath)






if __name__ == "__main__":
    tests = InitializationTest()
    # TASK 2
    #tests.kmeansInitTest()
    #tests.tissueModelInitTest()
    #tests.labelPropInitTest()

    # TASK 3
    tests.atlasAfterBestInitTest()
    #tests.atlasAfterLabelAndTissueInitTest()
    #tests.atlasIntoBestInitTest()
    #tests.atlasIntoLabelAndTissueInitTest()
