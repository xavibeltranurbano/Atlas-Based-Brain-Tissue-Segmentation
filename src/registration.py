# # -----------------------------------------------------------------------------
# # Tissue Models Segmentation File
# # Author: Xavier Beltran Urbano and Fredderik Hartmann
# # Date Created: 09-11-2023
# # -----------------------------------------------------------------------------


import itk # itk-elastix
import numpy as np
import os
import gc
import nibabel as nib

from utils import Utils

class Registration:

    util = Utils()

    def __init__(self, parameterFolder):
        self.parameterObject, self.registrationTypeList = self.initParamaterObject(parameterFolder)

    def initParamaterObject(self, parameterFolder):
        # initializes a parameter object with the registration in the parameter Folder. Automatically sorted after rigid -> affine -> bspline
        parameterObject = itk.ParameterObject.New()
        parameterMaps = self.util.getAllFiles(parameterFolder)
        sortedMaps = sorted(parameterMaps, key=self.util.getRegistrationSortKey)

        registrationTypeList = []
        for parameterPath in sortedMaps:
            parameterObject.AddParameterFile(parameterPath)
            registrationTypeList.append(os.path.basename(parameterPath).split(".")[0])
        return parameterObject, registrationTypeList

    def register(self, fixedImagePath, meanImagePath):
        # registers an image
        fixedImage = self.util.loadImageFrom(fixedImagePath)
        meanImage = self.util.loadImageFrom(meanImagePath)
        resultImage, resultTransformParameters = itk.elastix_registration_method(fixedImage, meanImage, parameter_object=self.parameterObject, log_to_console=False)

        registeredVolume = itk.transformix_filter(
                meanImage,
                transform_parameter_object=resultTransformParameters
        )

        registeredAtlas = self.applyTransform(fixedImagePath, resultTransformParameters)

        self.safeTransformParameterObject(resultTransformParameters, fixedImagePath)
        self.safeImage(resultImage, fixedImagePath)
        #self.safeAtlas(registeredAtlas, fixedImagePath)

    def applyTransform(self, fixedImagePath, resultTransformParameters):
        # applies the transformation to the atlas
        tst, affine = self.util.readNiftiImage(fixedImagePath)

        atlases = []
        for i in range(1,4):
            atlas = self.util.loadImageFrom(f"models/atlas{i}.nii.gz")

            registeredVolume = itk.transformix_filter(
                atlas,
                transform_parameter_object=resultTransformParameters
            )
            atlases.append(registeredVolume[..., np.newaxis]) 

        registeredAtlas = np.concatenate(atlases, axis=-1)
        reorderedImage = np.transpose(registeredAtlas, (2, 1, 0, 3))


        # store
        newImage = nib.Nifti1Image(reorderedImage,affine)
        imageName = os.path.basename(fixedImagePath).split(".")[0]
        imageName = "atlas_regTo_" + imageName + ".nii.gz"

        #folderPath = "registeredAtlases"
        folderPath = "MNIregisteredAtlases"

        self.util.ensureFolderExists(folderPath)
        storePath = os.path.join(folderPath, imageName)
        newImage.to_filename(storePath)
        print(storePath)


        return registeredAtlas


    def safeTransformParameterObject(self, resultTransformParameters, fixedImagePath):
        # saves the computed registration parameter file
        nParameterMaps = resultTransformParameters.GetNumberOfParameterMaps()
        folderPath = "transformationMatrices"

        self.util.ensureFolderExists(folderPath)
        imageName = os.path.basename(fixedImagePath)
        imageName = imageName.split(".")[0]

        for index in range(nParameterMaps):       
            fileName = imageName + "_" + self.registrationTypeList[index] + ".txt"
            finalPath = os.path.join(folderPath, fileName)
            parameterMap = resultTransformParameters.GetParameterMap(index)

            if index == nParameterMaps - 1:
                parameterMap['FinalBSplineInterpolationOrder'] =  "0"
            print(finalPath)
            self.parameterObject.WriteParameterFile(parameterMap, finalPath)
    
    def safeImage(self, image, fixedImagePath):
        # saves images in a nii.gz
        imageName = os.path.basename(fixedImagePath).split(".")[0]
        imageName = "meanImage_regTo_" + imageName + ".nii.gz"

        folderPath = "registeredImages"
        self.util.ensureFolderExists(folderPath)
        storePath = os.path.join(folderPath, imageName)
        itk.imwrite(image,storePath)
        print(storePath)
    
    def safeAtlas(self, atlas, fixedImagePath):
        # saves a registered atlas
        imageName = os.path.basename(fixedImagePath).split(".")[0]
        imageName = "atlas_regTo_" + imageName + ".nii.gz"

        folderPath = "registeredAtlases"
        self.util.ensureFolderExists(folderPath)
        storePath = os.path.join(folderPath, imageName)
        itk.imwrite(atlas,storePath)
        print(storePath)




if __name__ == "__main__":
    util = Utils()

    testFolder = "images/testing-images"
    paramterFolder = "Par0038"

    meanImagePath = "models/meanImage.nii.gz"
    fixedImagePaths = util.getAllFiles(testFolder)


    # registers all moving images
    for fixedImagePath in fixedImagePaths:
        print(f"calculating files for {os.path.basename(fixedImagePath)}")
        reg = Registration(paramterFolder)
        reg.register(fixedImagePath, meanImagePath)

        del reg
        gc.collect()



    
        