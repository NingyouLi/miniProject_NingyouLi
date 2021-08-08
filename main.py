import utils

def main():
    # read image
    img = sitk.ReadImage('Case11.mhd')
    goldenStandard_img = sitk.ReadImage('Case11_segmentation.mhd')
    # segmentation
    filtered_img = prostate_segmenter(img)
    # save segmentation
    saveSegmentation(filtered_img)
    # visulize overlay images
    print('my segmentation:')
    overlay_visulize(img, filtered_img)
    print('golden standard:')
    overlay_visulize(img, goldenStandard_img)
    # dice evaluations
    print('my segmentation:')
    seg_eval_dice(filtered_img, img)
    print('golden standard:')
    seg_eval_dice(goldenStandard_img, img)
    # hausdorff evaluations
    print('my segmentation:')
    seg_eval_hausdorff(filtered_img, img)
    print('golden standard:')
    seg_eval_hausdorff(goldenStandard_img, img)
    # find centroid point
    centroid_pt = get_target_loc(goldenStandard_img)
    # plot red X on image
    overlay_target(centroid_pt, img)
    # extract pixel intensity array
    result = pixel_extract(img, centroid_pt, 6)
    # boxplot of pixel intensity
    plotBoxplot(result, centroid_pt)


main()