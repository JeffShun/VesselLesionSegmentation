import numpy as np
import SimpleITK as sitk
import os


def cal_dice(pred, label):
    intersection = (pred*label).sum()
    union = pred.sum() + label.sum()
    dice = 2 * intersection / (union + 1e-7)
    return dice

def save_result_as_image(images, preds, labels, save_dir):
    for i in range(images.shape[0]):
        img = (images[i,0].cpu().numpy()*255).astype("uint8")
        label = labels[i,0].cpu().numpy().astype("int32")
        pred = (preds[i,0].cpu().numpy() > 0.5).astype("int32")

        dice = cal_dice(pred, label)

        pred_itk = sitk.GetImageFromArray(pred)
        label_itk = sitk.GetImageFromArray(label)
        img_itk = sitk.GetImageFromArray(img)

        sitk.WriteImage(pred_itk, os.path.join(save_dir, f'{i+1}-dice{dice:.2f}.seg.nii.gz'))
        sitk.WriteImage(label_itk, os.path.join(save_dir, f'{i+1}.label.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(save_dir, f'{i+1}.dcm.nii.gz'))


