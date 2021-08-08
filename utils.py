# https://github.com/NingyouLi/miniProject_NingyouLi.git2
# import packages
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def prostate_segmenter(image, glambda=0.2, seed_pts=(259,253,22), TimeStep=0.125, conductance=0.3, sigma=0.1, \
                       alpha=-0.3, beta=2.0, stoppingValue=480, upperThreshold=800, lowerThreshold=0):
    """
    :param image: image object, input image
    :param glambda: lambda, parameter for gamma filtering
    :param seed_pts: a point inside the object you wish to segmented
    :param TimeStep: parameter of CurvatureAnisotropicDiffusionImageFilter
    :param conductance: parameter of CurvatureAnisotropicDiffusionImageFilter
    :param sigma: parameter of gradient filter, representing the size of the mask
    :param alpha: parameter of sigmoid filter
    :param beta: parameter of sigmoid filter
    :param stoppingValue: parameter of Fast Marching filter
    :param upperThreshold: set the upper threshold for thresholding
    :param lowerThreshold: set the lower threshold for thresholding
    :return: the segmented image
    """
    slice_no = round(image.GetSize()[2]/2)
    img = sitk.Cast(image, sitk.sitkFloat32)
    # brighten the image using gamma filter to show more details in dark
    filtered_img = ((image/255)**glambda) # lambda<0
    gamma_img = sitk.RescaleIntensity(filtered_img,0,255)
    # smoothing filter
    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(TimeStep)
    smoothing.SetConductanceParameter(conductance)
    smoothingOutput = smoothing.Execute(gamma_img)
    # gradient filter
    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(sigma)
    gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)
    # sigmoid curve filter
    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    sigmoid.DebugOn()
    sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)
    # fast marching
    fastMarching = sitk.FastMarchingImageFilter()
    #trialPoint = (259,253,22)
    fastMarching.AddTrialPoint(seed_pts)
    fastMarching.SetStoppingValue(stoppingValue)
    fastMarchingOutput = fastMarching.Execute(sigmoidOutput)
    # thresholding
    thresholder = sitk.BinaryThresholdImageFilter()
    thresholder.SetLowerThreshold(lowerThreshold)
    thresholder.SetUpperThreshold(upperThreshold)
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(255)
    result_img = thresholder.Execute(fastMarchingOutput)

    plt.figure(figsize=(10,5))
    plt.imshow(sitk.GetArrayFromImage(result_img[:,:,slice_no]))
    return result_img

def saveImage(image, fileName='my_segmentation.nrrd'):
    """
    :param image: input image
    :param fileName: the filename for the output .nrrd file
    :return: None
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fileName)
    writer.Execute(image)
    return None

def overlay_visulize(image, mask_img):
    """
    :param image: input image
    :param mask_img: segmented image
    :return: the overlay image
    """
    slice_no = round(image.GetSize()[2]/2)
    overlay_image = sitk.LabelOverlay(image[:,:,slice_no], mask_img[:,:,slice_no])
    plt.figure(figsize=(10,5))
    plt.imshow(sitk.GetArrayFromImage(overlay_image).astype(np.uint8))
    plt.show()
    return overlay_image

def seg_eval_dice(filtered_img, img):
    """
    :param filtered_img: segmented image
    :param img: image object
    :return: DSC of the filtered image and the original image
    """
    dice_dist = sitk.LabelOverlapMeasuresImageFilter()
    dice_dist.Execute(img>0.5, filtered_img>0.5)
    dice = dice_dist.GetDiceCoefficient()
    print('Dice evaluation: ', dice)
    return dice


def seg_eval_hausdorff(filtered_img, img):
    """
    :param filtered_img: filtered image object
    :param img: image object
    :return: the HD of the filtered image and the original image object
    """
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(img>0.5, filtered_img>0.5)
    #AverageHD = hausdorffcomputer.GetAverageHausdorffDistance()
    HD = hausdorffcomputer.GetHausdorffDistance()
    print('HausdorffDistance:', HD)
    return HD

def findMaxArea(seg_img):
    """
    :param seg_img: segmented image object (golden standard image in this case)
    :return: the slice with maximum segmented area
    """
    area_lst = []
    for slice_no in range(seg_img.GetSize()[2]):
        img_array = sitk.GetArrayFromImage(seg_img[:,:,slice_no])
        pixels = len(np.column_stack(np.where(img_array > 0)))
        # Calculate the area of the prostate
        new_area = (seg_img[:,:,slice_no].GetSpacing()[0]*seg_img[:,:,slice_no].GetSpacing()[1])*pixels
        area_lst.append(new_area)
    slices = area_lst.index(max(area_lst))
    print('max area of slide', slices, 'is', max(area_lst))
    return slices

def get_target_loc(seg_img):
    """
    :param seg_img: the segmented image object
    :return: the physical coordinate of the biopsy target
    """
    slide_no = findMaxArea(seg_img)
    segImage_array = sitk.GetArrayFromImage(seg_img[:,:,slide_no])
    x = np.mean(np.column_stack(np.where(segImage_array > 0)[1]))
    y = np.mean(np.column_stack(np.where(segImage_array > 0)[0]))
    print('centroid in index:', (x,y,slide_no))
    physical = seg_img.TransformContinuousIndexToPhysicalPoint((x,y,slide_no))
    print('centroid in actual dimension:', physical)
    return physical

def overlay_target(coordinate, image):
    """
    :param coordinate: the physical coordinate of the red X
    :param image: the input image object
    :return:None
    """
    slide_no = findMaxArea(image)
    image_array = sitk.GetArrayFromImage(image[:,:,slide_no])
    pixel_coor = image.TransformPhysicalPointToIndex(coordinate)
    plt.imshow(image_array, cmap='gray')
    plt.annotate('X', xy=(pixel_coor[0],pixel_coor[1]),color= 'red')
    plt.show()
    return None


def pixel_extract(image, point, width):
    """
    :param image: image object
    :param point: physical coordinate of the centre of the cube
    :param width: The width of the cube
    :return: a list of pixel intensities
    """
    sub = (image.TransformPhysicalPointToIndex((point[0]-width/2, point[1]-width/2, point[2]-width/2)))
    add = (image.TransformPhysicalPointToIndex((point[0]+width/2, point[1]+width/2, point[2]+width/2)))
    slide_lst = list(range(sub[2], (add[2])+1))
    pixel_lst = []
    for slide_no in slide_lst:
        slide_array = sitk.GetArrayFromImage(image[sub[0]:add[0], sub[1]:add[1], slide_no])
        pixel_lst.append(slide_array.flatten())
    flat_list = [item for sublist in pixel_lst for item in sublist]
    print('\nend of pixel extacting')
    return flat_list


def plotBoxplot(data, point):
    """
    :param data: list, a list of pixel intensities
    :param point: the physical coordinate of the centre of the cube
    :return: None
    """
    plt.boxplot(data)
    plt.xlabel(str(point))
    plt.ylabel('Pixel Intensity')
    plt.show()
    return None