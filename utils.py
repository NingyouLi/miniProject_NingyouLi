# import packages
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def prostate_segmenter(image, glambda=0.2, seed_pts=[(256, 252, 22)], Multiplier=1.31):
    slice_no = round(image.GetSize()[2] / 2)
    img = sitk.Cast(image, sitk.sitkFloat32)
    # brighten the image using gamma filter to show more details in dark
    filtered_img = ((image / 255) ** glambda)  # lambda<0
    gamma_img = sitk.RescaleIntensity(filtered_img, 0, 255)

    # Use region growing to segment the prostate
    growing_filter = sitk.ConfidenceConnectedImageFilter()
    growing_filter.SetSeedList(seed_pts)
    # set range of pixel intensity
    growing_filter.SetMultiplier(Multiplier)
    growed_img = growing_filter.Execute(gamma_img)
    plt.figure(figsize=(10, 5))
    plt.imshow(sitk.GetArrayFromImage(growed_img[:, :, slice_no]))
    return growed_img

def saveSegmentation(growed_image):
    writer = sitk.ImageFileWriter()
    writer.SetFileName('my_segmentation.nrrd')
    writer.Execute(growed_image)
    return None

def overlay_visulize(image, mask_img):
    slice_no = round(image.GetSize()[2]/2)
    overlay_image = sitk.LabelOverlay(image[:,:,slice_no], mask_img[:,:,slice_no])
    plt.figure(figsize=(10,5))
    plt.imshow(sitk.GetArrayFromImage(overlay_image).astype(np.uint8))
    plt.show()
    return None

def seg_eval_dice(filtered_img, img):
    dice_dist = sitk.LabelOverlapMeasuresImageFilter()
    dice_dist.Execute(img>0.5, filtered_img>0.5)
    dice = dice_dist.GetDiceCoefficient()
    print('Dice evaluation: ', dice)
    return dice


def seg_eval_hausdorff(filtered_img, img):
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(img>0.5, filtered_img>0.5)
    #AverageHD = hausdorffcomputer.GetAverageHausdorffDistance()
    HD = hausdorffcomputer.GetHausdorffDistance()
    print('HausdorffDistance:', HD)
    return HD

def findMaxArea(seg_img):
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
    slide_no = findMaxArea(seg_img)
    segImage_array = sitk.GetArrayFromImage(seg_img[:,:,slide_no])
    x = np.mean(np.column_stack(np.where(segImage_array > 0)[1]))
    y = np.mean(np.column_stack(np.where(segImage_array > 0)[0]))
    print('centroid in index:', (x,y,slide_no))
    physical = seg_img.TransformContinuousIndexToPhysicalPoint((x,y,slide_no))
    print('centroid in actual dimension:', physical)
    return physical

def overlay_target(coordinate, image):
    slide_no = findMaxArea(image)
    image_array = sitk.GetArrayFromImage(image[:,:,slide_no])
    pixel_coor = image.TransformPhysicalPointToIndex(coordinate)
    plt.imshow(image_array, cmap='gray')
    plt.annotate('X', xy=(pixel_coor[0],pixel_coor[1]),color= 'red')
    plt.show()
    return None


def pixel_extract(image, point, width):
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
    plt.boxplot(data)
    plt.xlabel(str(point))
    plt.ylabel('Pixel Intensity')
    plt.show()
    return None