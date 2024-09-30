import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
import torch
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones import ResUNet3D
from custom.model.loss import MixLoss
from custom.model.network import Segmentation_Network

class stenosis_segmentation_cfg:

    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'
    
    # img
    img_size = (256, 256, 256)
    
    # network
    network = Segmentation_Network(
        backbone = ResUNet3D(
            in_ch = 2,
            channels = 16, 
            blocks = 2,
            aspp=True
            ),
        channels = 16,
        n_class = 1
    )

    # loss function
    loss_f = MixLoss(BinaryDiceLoss_weight=1.0, BCELoss_weight=1.0, SBCELoss_weight=0.0, Sensitivity_SpecificityLoss_weight=0.0, MaskLoss_weight=1.0)

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/stenosis/train.txt",
        gaussian_sigma = 0.5,
        transforms = TransformCompose([
            normlize(win_clip=True),
            to_tensor(use_gpu=False),
            # # resize(img_size),
            # random_gamma_transform(gamma_range=[0.8, 1.2], prob=0.5),
            # # random_flip(axis=1, prob=0.5),
            # # random_flip(axis=2, prob=0.5),
            # random_flip(axis=3, prob=0.5),
            # random_rotate3d(x_theta_range=[-20,20],
            #                 y_theta_range=[-20,20],
            #                 z_theta_range=[-20,20],
            #                 prob=0.5),
            # random_add_gaussian_noise(prob=0.5, mean=0, std=0.02),
            ])
        )
    
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/stenosis/valid.txt",
        gaussian_sigma = 0.5,
        transforms = TransformCompose([
            normlize(win_clip=None),
            to_tensor(),
            # resize(img_size)
            ])
        )
    
    # train dataloader
    batchsize = 2
    shuffle = True
    num_workers = 8
    drop_last = False

    # optimizer
    lr = 1e-3
    weight_decay = 5e-4

    # scheduler
    milestones = [20,40,80]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 1
    warmup_method = "linear"
    last_epoch = -1

    # debug
    total_epochs = 100
    valid_interval = 1
    checkpoint_save_interval = 1
    log_dir = work_dir + "/Logs/stenosis/ResUNet3D-RS-AUG-ASPP-Finetue"
    checkpoints_dir = work_dir + '/checkpoints/stenosis/ResUNet3D-RS-AUG-ASPP-Finetue'
    load_from = work_dir + '/checkpoints/stenosis/ResUNet3D-RS-AUG-ASPP/100.pth'
