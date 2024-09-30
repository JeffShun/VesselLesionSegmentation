import argparse
import os
import sys
import pickle
import itk
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import PredictModel, Predictor


def parse_args():
    parser = argparse.ArgumentParser(description='Test Segmentation3D')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--task', default='stenosis', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        default='../train/checkpoints/stenosis/ResUNet3D-RS-AUG-ASPP-Finetue/100.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def box2mask(boxes, mask_shape):
    boxes_mask = np.zeros(mask_shape)
    for box in boxes:
        x_min, y_min, z_min, x_max, y_max, z_max = box
        boxes_mask[x_min:x_max,y_min:y_max,z_min:z_max] = 1
    return boxes_mask

def parse_label_box(bboxes):
    bboxes_ = []
    for bbox in bboxes:
        bbox_ = list(bbox[0])
        bboxes_.append(bbox_)
    return np.array(bboxes_)


def main(device, task, args):
    input_path = f'./data/input/{task}'
    label_path = f'./data/label/{task}'
    box_path = f'./data/box/bbox_{task}.pkl'
    output_path = f'./data/output/{task}/ResUNet3D-RS-AUG-ASPP-Finetue'

    bbox_dic = pickle.load(open(box_path, "rb")) 

    # TODO: 适配参数输入
    model_segmentation = PredictModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_segmentation = Predictor(
        device=device,
        task=task,
        model=model_segmentation,
    )
    os.makedirs(output_path, exist_ok=True)

    for sample in tqdm(os.listdir(input_path)):  
        sitk_img = sitk.ReadImage(os.path.join(input_path, sample))
        pred, heatmap = predictor_segmentation.predict(sitk_img)
        label_sitk = sitk.ReadImage(os.path.join(label_path, sample.replace(".nii.gz","seg.nii.gz"))) 
        label = sitk.GetArrayFromImage(label_sitk)

        pred_sitk = sitk.GetImageFromArray(pred)
        pred_sitk.CopyInformation(sitk_img)
        heatmap_sitk = sitk.GetImageFromArray(heatmap)
        heatmap_sitk.CopyInformation(sitk_img)
        # vesselmask_itk = itk.GetImageFromArray(vesselmask)
        # vesselmask_itk.CopyInformation(itkimg)
        pid = sample.replace(".nii.gz","")
        sitk.WriteImage(sitk_img, os.path.join(output_path, f'{pid}.dcm.nii.gz'))
        sitk.WriteImage(heatmap_sitk, os.path.join(output_path, f'{pid}.heatmap.nii.gz'))
        sitk.WriteImage(pred_sitk, os.path.join(output_path, f'{pid}.seg.nii.gz'))
        sitk.WriteImage(label_sitk, os.path.join(output_path, f'{pid}.label.nii.gz'))
        # itk.imwrite(vesselmask_itk, os.path.join(output_path, f'{pid}.vesselmask.nii.gz'))
        
        data_for_metrics = os.path.join(output_path, "data_for_metrics")
        os.makedirs(data_for_metrics, exist_ok=True)
        np.savez_compressed(os.path.join(data_for_metrics, f'{pid}.npz'), pred=pred, label=label)



if __name__ == '__main__':
    args = parse_args()
    main(
        device=args.device,
        task=args.task,
        args=args,
    )