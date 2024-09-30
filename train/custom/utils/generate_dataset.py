"""生成模型输入数据."""

import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import pickle
from skimage.morphology import skeletonize_3d
from skimage.measure import label
import itk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    return args


def gen_lst(save_path):
    save_file = os.path.join(save_path, 'dataset.txt')
    data_list = glob.glob(os.path.join(save_path, '*.npz'))
    with open(save_file, 'w') as f:
        for i, data in enumerate(data_list):
            f.writelines(data.replace("\\","/") + '\n')
    print('num of data: ', i+1)


def bbox2arr(bboxes):
    bboxes_ = []
    for bbox in bboxes:
        bbox_ = list(bbox[0])
        bbox_.append(bbox[1])
        bboxes_.append(bbox_)
    return np.array(bboxes_)


def get_outer_box(mask_array):
    foreground_indices = np.argwhere(mask_array == 1)
    if foreground_indices.size == 0:
        raise ValueError("No foreground pixels found in the mask.")

    min_coords = foreground_indices.min(axis=0)
    max_coords = foreground_indices.max(axis=0) + 1 
    return [min_coords, max_coords]


def crop_and_pad(img, box_coords, target_size):
    min_coords, max_coords = box_coords
    # 裁剪图像
    cropped_img = img[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
        ]

    # 填充图像至固定尺度
    pad_size = [max(0, target_size[i] - cropped_img.shape[i]) for i in range(3)]
    pad_left = [pad_size[i] // 2 for i in range(3)]
    pad_right = [pad_size[i] - pad_left[i] for i in range(3)]
    padded_img = np.pad(cropped_img, 
                         pad_width=[(pad_left[0], pad_right[0]), 
                                    (pad_left[1], pad_right[1]), 
                                    (pad_left[2], pad_right[2])], 
                         mode='constant', constant_values=0)

    return padded_img


def resample_image(image, new_spacing):
    image = sitk.Cast(image, sitk.sitkFloat32)
    
    # 获取原始 spacing 和尺寸
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新的尺寸
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(len(new_spacing))]

    # 创建 resample 参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear) 

    # 执行 resample 操作
    resampled_image = resampler.Execute(image)

    return resampled_image


def get_crop_area(mask, crop_size):
    # 获取外接立方体的边界框
    z_min, y_min, x_min = np.min(np.nonzero(mask), axis=1)
    z_max, y_max, x_max = np.max(np.nonzero(mask), axis=1)

    # 计算外接立方体的中心
    center_of_mass = np.array([(z_min + z_max) // 2,
                               (y_min + y_max) // 2,
                               (x_min + x_max) // 2])
    
    # 获取裁剪区域的尺寸
    depth, height, width = crop_size

    # 计算裁剪区域的边界
    start_z = max(z_max - depth, 0)
    start_y = max(center_of_mass[1] - height // 2, 0)
    start_x = max(center_of_mass[2] - width // 2, 0)
    
    end_z = min(start_z + depth, mask.shape[0])
    end_y = min(start_y + height, mask.shape[1])
    end_x = min(start_x + width, mask.shape[2])

    # 调整起始位置以适应边界条件
    start_y = end_y - depth if end_y == mask.shape[1] else start_y
    start_x = end_x - width if end_x == mask.shape[2] else start_x

    crop_area = np.zeros_like(mask, dtype=np.uint8)
    crop_area[start_z:end_z, start_y:end_y, start_x:end_x] = 1

    return crop_area


def get_longest_components(mask, num_components):
    # 使用连通域分析找出最长的连通区域
    labeled_mask = label(mask, connectivity=3) 
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # 背景的体积不计入
    component_volumes = [(i, sizes[i]) for i in range(1, len(sizes)) if sizes[i] > 0]
    # 按体积排序并选取前 num_components 个
    longest_components = sorted(component_volumes, key=lambda x: x[1], reverse=True)[:num_components]
    # print(longest_components)
    largest_component_indices = {i[0] for i in longest_components if i[1] > 6000}
    
    # 只保留这些最大的连通区域
    filtered_mask = np.isin(labeled_mask, list(largest_component_indices)).astype(np.uint8)
    return filtered_mask
    

def vesselSegment(input_image):

    Dimension = input_image.GetImageDimension()
    ImageType = type(input_image)

    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(0.5)
    objectness_filter.SetGamma(2.0)
    
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(0.1)
    multi_scale_filter.SetSigmaMaximum(5.0)
    multi_scale_filter.SetNumberOfSigmaSteps(6)
    
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    
    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter.GetOutput())

    enhanced_img = itk.GetArrayFromImage(rescale_filter.GetOutput())

    # enhanced_itk = itk.GetImageFromArray(enhanced_img)
    # enhanced_itk.CopyInformation(input_image)
    # itk.imwrite(enhanced_itk, './enhanced_img.nii.gz')

    crop_area = get_crop_area(enhanced_img > 80, [256, 256, 256])

    vessel_mask = (enhanced_img > 20) * crop_area
    vessel_mask = get_longest_components(vessel_mask, 6)
    
    # mask_itk = itk.GetImageFromArray(vessel_mask.astype("uint8"))
    # mask_itk.CopyInformation(input_image)
    # itk.imwrite(mask_itk, './mask.nii.gz')

    return vessel_mask.astype("uint8")


if __name__ == '__main__':
    args = parse_args()
    threads = args.threads
    data_path = args.data_path
    save_path = args.save_path
    target_spacing = [0.45,0.45,0.45]
    target_size = [256, 256, 256]
    os.makedirs(save_path, exist_ok=True)
    
    print("\nBegin to generate healthy training dataset!")
    healthy_sample_dir = os.path.join(data_path, "healthy_training")
    for healthy_sample in tqdm(os.listdir(healthy_sample_dir)):
        pid = healthy_sample.replace(".nii.gz","")
        img_path = os.path.join(data_path, "healthy_training", f"{pid}.nii.gz")  
        sitk_img = sitk.ReadImage(img_path)
        sitk_img = resample_image(sitk_img, target_spacing)
        img = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        itk_img = itk.GetImageFromArray(img)
        vesselmask = vesselSegment(itk_img)
        
        seg = np.zeros_like(img)
        bbox = np.array([[]])
        
        # crop vesselmask以外的区域
        crop_coords = get_outer_box(vesselmask)
        img_patch = crop_and_pad(img, crop_coords, target_size)
        seg_patch = crop_and_pad(seg, crop_coords, target_size)
        vesselmask_patch = crop_and_pad(vesselmask, crop_coords, target_size)
        print(img_patch.shape, seg_patch.shape, vesselmask_patch.shape)
        np.savez_compressed(os.path.join(save_path, f'healthy{pid}.npz'), img=img_patch, vesselmask=vesselmask_patch, seg=seg_patch, bbox=bbox)


    print("\nBegin to generate stenosis training dataset!")
    stenosis_sample_dir = os.path.join(data_path, "stenosis_training")
    stenosis_bbox_dic = pickle.load(open(os.path.join(data_path, "bbox_stenosis.pkl"), "rb")) 
    for stenosis_sample in tqdm(os.listdir(stenosis_sample_dir)):
        pid = stenosis_sample.replace(".nii.gz","")
        img_path = os.path.join(data_path, "stenosis_training", f"{pid}.nii.gz")  
        sitk_img = sitk.ReadImage(img_path)
        sitk_img = resample_image(sitk_img, target_spacing)
        img = sitk.GetArrayFromImage(sitk_img)
        itk_img = itk.GetImageFromArray(img)
        vesselmask = vesselSegment(itk_img)
        
        seg_path = os.path.join(data_path, "stenosis_training_seg", f"{pid}seg.nii.gz")  
        sitk_seg = sitk.ReadImage(seg_path)
        sitk_seg = resample_image(sitk_seg, target_spacing)
        sitk.WriteImage(sitk_seg, "./temp.nii.gz")

        seg = (sitk.GetArrayFromImage(sitk_seg) > 0.3).astype("uint8")
        bbox = bbox2arr(stenosis_bbox_dic[f"{pid}.nii.gz"])    

        # crop vesselmask以外的区域
        crop_coords = get_outer_box(vesselmask)
        img_patch = crop_and_pad(img, crop_coords, target_size)
        seg_patch = crop_and_pad(seg, crop_coords, target_size)
        vesselmask_patch = crop_and_pad(vesselmask, crop_coords, target_size)
        print(img_patch.shape, seg_patch.shape, vesselmask_patch.shape, (vesselmask_patch*seg_patch).sum())
        np.savez_compressed(os.path.join(save_path, f'stenosis{pid}.npz'), img=img_patch, vesselmask=vesselmask_patch, seg=seg_patch, bbox=bbox)  


    print("\nBegin to generate IA training dataset!")
    IA_sample_dir = os.path.join(data_path, "IA_training")
    IA_bbox_dic = pickle.load(open(os.path.join(data_path, "bbox_IA.pkl"), "rb"))  
    for IA_sample in tqdm(os.listdir(IA_sample_dir)):
        pid = IA_sample.replace(".nii.gz","")
        img_path = os.path.join(data_path, "IA_training", f"{pid}.nii.gz")  
        sitk_img = sitk.ReadImage(img_path)
        sitk_img = resample_image(sitk_img, target_spacing)
        img = sitk.GetArrayFromImage(sitk_img)
        itk_img = itk.GetImageFromArray(img)
        vesselmask = vesselSegment(itk_img)
       
        seg_path = os.path.join(data_path, "IA_training_seg", f"{pid}seg.nii.gz")  
        sitk_seg = sitk.ReadImage(seg_path)
        sitk_seg = resample_image(sitk_seg, target_spacing)
        seg = (sitk.GetArrayFromImage(sitk_seg) > 0.3).astype("uint8")
        bbox = bbox2arr(IA_bbox_dic[f"{pid}.nii.gz"])   

        # crop vesselmask以外的区域
        crop_coords = get_outer_box(vesselmask)
        img_patch = crop_and_pad(img, crop_coords, target_size)
        seg_patch = crop_and_pad(seg, crop_coords, target_size)
        vesselmask_patch = crop_and_pad(vesselmask, crop_coords, target_size)
        print(img_patch.shape, seg_patch.shape, vesselmask_patch.shape, (vesselmask_patch*seg_patch).sum())
        np.savez_compressed(os.path.join(save_path, f'IA{pid}.npz'), img=img_patch, vesselmask=vesselmask_patch, seg=seg_patch, bbox=bbox)  

        """
        # Just for Debug
        box_mask = np.zeros_like(img)
        z_min, y_min, x_min, z_max, y_max, x_max, label = bbox[0]
        # SimpleITK读取的array各轴顺序为[z,y,x]
        box_mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1
        box_itk = sitk.GetImageFromArray(box_mask)
        box_itk.CopyInformation(itkimg)
        debug_dir = "./train_data/debug"
        os.makedirs(debug_dir, exist_ok=True)
        sitk.WriteImage(itkimg, os.path.join(debug_dir, f'IA{pid}.dcm.nii.gz'))
        sitk.WriteImage(box_itk, os.path.join(debug_dir, f'IA{pid}.box.nii.gz'))
        """
    # 生成Dataset所需的数据列表
    gen_lst(save_path)


    