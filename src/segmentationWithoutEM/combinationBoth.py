# # -----------------------------------------------------------------------------
# # Combination of Both (Tissue Models and Label Propagation ) Segmentation File
# # Author: Xavier Beltran Urbano and Fredderik Hartmann
# # Date Created: 09-11-2023
# # -----------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
from src.utils import Utils
import pandas as pd
import time


class TissueModelsSegmentation:
    util = Utils()

    def __init__(self, datasetFolder):
        self.datasetPath = datasetFolder
        self.imagePaths = os.path.join(datasetFolder, 'testing-images')
        self.gtPaths = os.path.join(datasetFolder, 'testing-labels')
        self.affineParams = []
        self.shapeImages = []
        self.diceSegmentation = {'WM': [], 'GM': [], 'CSF': []}
        self.atlasPaths = np.sort(self.util.getAllFiles('registeredAtlases'))
        self.metrics = {'WM': [], 'GM': [], 'CSF': [], 'Time': []}

    def loadAtlas(self,atlasPath):
        # loads the atlas
        atlas, _ = self.util.readNiftiImage(atlasPath)
        return atlas

    def execute(self):
        # exectues the segmentation
        vecImages, vecGT = self.readDataset()
        self.tissueModels = pd.read_csv('models/TissueModel.csv')
        self.LabelPropagationList = [self.loadAtlas(atlasPath) for atlasPath in self.atlasPaths]
        self.startTime = time.time()
        self.segmentImages(vecImages, vecGT)

    ######################################################
    ############### Read Dataset and Normalizing #########
    ######################################################

    def readDataset(self):
        # reads the dataset (images, metadata, groundtrouth), return a vector of images and a vector matching GT
        file_names = np.sort([file.split('.')[0] for file in os.listdir(self.imagePaths)])
        vecImages = []
        vecGT = []
        for file in file_names:
            image, _ = self.util.readNiftiImage(os.path.join(self.imagePaths, file +'.nii.gz'))
            gt, affine = self.util.readNiftiImage(os.path.join(self.gtPaths, file +'_3C.nii.gz'))
            # Preprocess image
            maskedImage = image[gt > 0]
            image=np.where(gt > 0,image,0)
            image[gt > 0] = self.normalizeImage(maskedImage)
            # Append the corresponding images/params to their list
            self.affineParams.append(affine),vecImages.append(image.astype(np.uint8)), vecGT.append(gt)

        return vecImages, vecGT

    @staticmethod
    def normalizeImage(vec, newMin=0, newMax=255):
        # min max normalization
        minVal = np.min(vec)
        maxVal = np.max(vec)
        normalized = (vec - minVal) / (maxVal - minVal) * (newMax - newMin) + newMin
        return normalized

    ######################################################################################################
    ### Multiply Probabilities, Segment Image and Compute Metrics ###############################
    ######################################################################################################

    def applyMultiplication(self, image,LabelPropagation):
        segImage = np.zeros(LabelPropagation.shape)
        for channelTissue in range(LabelPropagation.shape[3]):
                for i in range(256):
                    segValue = self.tissueModels.iloc[i, channelTissue]
                    segImage[...,channelTissue][image == i] = segValue
                segImage[...,channelTissue]=(segImage[...,channelTissue] * LabelPropagation[..., channelTissue])
        finalSeg = np.argmax(segImage, axis=3)+1
        binaryImage = np.where(image > 0, 1, 0)
        return finalSeg*binaryImage

    def computeMetrics(self, segmentedImage, gt, timeDiff):
        # computes metrics: dice and stores computation time
        self.metrics['WM'].append(self.util.diceCoefficient(segmentedImage == 1, gt == 1))
        self.metrics['CSF'].append(self.util.diceCoefficient(segmentedImage == 2, gt == 2))
        self.metrics['GM'].append(self.util.diceCoefficient(segmentedImage == 3, gt == 3))
        self.metrics['Time'].append(timeDiff)

    def segmentImages(self, vecImages, vecGT):
        # Segment the Images using the Tissue Models Probabilities and probabalistic atlas
        for i, image in enumerate(vecImages):
            print(i)
            startTime = time.time()
            segmentedImage = self.applyMultiplication(image,self.LabelPropagationList[i])
            self.computeMetrics(segmentedImage, vecGT[i],time.time()-startTime)
            utils.save_image(segmentedImage.astype(np.uint8), i,"Results/CombinationBoth", self.affineParams[i])
        self.saveResults('ResultsCombination.csv')

    ################################################
    ################ Save Results ##################
    ################################################
    def saveResults(self, fileName):
        # saves results in a csv file
        dice_df = pd.DataFrame(self.metrics, columns=['WM', 'GM', 'CSF', 'Time'])
        mean_values = dice_df.mean()  # Last row contains the mean of each column
        dice_df = dice_df._append(mean_values, ignore_index=True)
        dice_df.to_csv('Results/CombinationBoth/' + fileName, index=False)


if __name__ == "__main__":
    utils = Utils()
    generalPath = 'images'
    tissueModels = TissueModelsSegmentation(generalPath)
    tissueModels.execute()


