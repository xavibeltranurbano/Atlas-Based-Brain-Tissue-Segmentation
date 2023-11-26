# -----------------------------------------------------------------------------
# Utils File
# Author: Xavier Beltran Urbano AND Fredderik Hartmann
# Date Created: 09-11-2023
# -----------------------------------------------------------------------------



import itk
import os
import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment

class Utils:
    def __init__self(self):
        pass

    @staticmethod
    def loadImageFrom(imagePath):
        # Load images with itk floats (itk.F). Necessary for elastix
        return itk.imread(imagePath, itk.F)

    @staticmethod
    def save_image(img, case,path, affine):
        # Save the images in Nifti format
        nii_image = nib.Nifti1Image(img, affine=affine)  # Replace affine with the appropriate transformation matrix
        file_path = os.path.join(path, str(case) +".nii.gz")

        # Check for write permissions
        if os.access(os.path.dirname(file_path), os.W_OK):
            try:
                nib.save(nii_image, file_path)
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")
        else:
            raise PermissionError(f"No write access to the directory: {os.path.dirname(file_path)}")

    @staticmethod
    def loadTransformParameterObject(filePaths):
        # initializes 
        parameterObject = itk.ParameterObject.New()
        for parameterPath in filePaths:
            parameterObject.AddParameterFile(parameterPath)
        return parameterObject

    @staticmethod
    def getAllFiles(folderPath):
        allImagePaths = os.listdir(folderPath)
        relativePaths = []
        for imagePath in allImagePaths:
            if ".DS_Store" in imagePath:
                continue
            else:
                newPath = os.path.join(folderPath, imagePath)
                relativePaths.append(newPath)
        return relativePaths

    def splitFixedFromMoving(self, relativePaths, fixedName):
        fixedImageIndex = self.getImageIndex(relativePaths, fixedName)
        fixedImagePath = relativePaths.pop(fixedImageIndex)
        return fixedImagePath, relativePaths

    @staticmethod
    def loadAllImagesFromList(self, pathList):
        # loads all images from pathList
        allImages = []
        for imagePath in pathList:
            allImages.append(self.util.loadImageFrom(imagePath))
        return allImages

    @staticmethod
    def getImageIndex(relativePaths, imageName):
        index = 0
        for relativePath in relativePaths:
            if imageName in relativePath:
                return index
            else:
                index += 1
        raise ValueError(f"{imageName} not found in {relativePaths}")

    def readNiftiImage(self, filePath):
        # Read Nifti image
        try:
            niftiImage = nib.load(filePath).get_fdata()
            return niftiImage, nib.load(filePath).affine
        except Exception as e:
            print(f"Error reading NIFTI image from {filePath}: {str(e)}")

    @staticmethod
    def getRegistrationSortKey(filePath, fallbackSortValue = 99):
        # first sorted by registration type, than by filename
        primarySortOrder = {
            'rigid': 1,
            'affine': 2,
            'bspline': 3
        }
        primarySortValue = fallbackSortValue 
        fileName = os.path.basename(filePath)  
        secondarySortValue = fileName
        
        for keyword, orderValue in primarySortOrder.items():
            if keyword in fileName:
                primarySortValue = orderValue
                break
        return (primarySortValue, secondarySortValue)

    @staticmethod
    def ensureFolderExists(folderPath):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

    @staticmethod
    def reconstruct_image(cluster_assignments, GT):
        # Reconstruct segmentation to the original shape
        reconstruct_image = np.zeros(GT.shape)
        counter = 0
        for i in range(GT.shape[0]):
            for j in range(GT.shape[1]):
                for k in range(GT.shape[2]):
                    if GT[i, j, k] > 0:
                        reconstruct_image[i, j, k] = cluster_assignments[counter] + 1
                        counter += 1
        return reconstruct_image

    def fitLabelToGT(self, segmentation, gt, n_classes):
        # Re-assign labels using the Dice score with Hungarian algorithm

        matchingImage = np.zeros(gt.shape)
        costMatrix = np.zeros((n_classes+1, n_classes+1))

        for GTCluster in range(0, n_classes + 1):
            dices = []
            for Segmentationcluster in range(n_classes + 1):
                segBinary = np.where(segmentation == Segmentationcluster, 1, 0)
                gtBinary = np.where(gt == GTCluster, 1, 0)

                dice = self.diceCoefficient(segBinary, gtBinary)
                costMatrix[GTCluster][Segmentationcluster] = 1 - dice

        gt_indices, seg_indices = linear_sum_assignment(costMatrix)

        for i in range(0, n_classes+1):
            gtIdx = gt_indices[i]
            segIdx = seg_indices[i]
            matchingImage[segmentation == segIdx] = gtIdx
        return matchingImage


    def diceCoefficient(self, segmentation_mask, GT_mask):
        # Compute the Dice score
        intersection = (segmentation_mask * GT_mask).sum()
        total_area = segmentation_mask.sum() + GT_mask.sum()
        dice = (2.0 * intersection) / total_area
        return dice

    def loadAtlasCorrespondingTo(self, imagePath,atlasPaths):
        imageName = os.path.basename(imagePath).split(".")[0]
        for atlasPath in atlasPaths:
            atlasName = os.path.basename(atlasPath).split("_")[2].split(".")[0]
            if atlasName in imageName:
                atlas, _ = self.utils.readNiftiImage(atlasPath)
                return atlas
        raise FileNotFoundError(f"No matching atlas file for: {imagePath} in registeredAtlases found.")


    @staticmethod
    def normalizeImage(vec, newMin=0, newMax=255):
        # min max normalization
        minVal = np.min(vec)
        maxVal = np.max(vec)
        normalized = (vec - minVal) / (maxVal - minVal) * (newMax - newMin) + newMin
        return normalized
