import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import shutil
from config.model_config_stenosis import stenosis_segmentation_cfg
from config.model_config_IA import IA_segmentation_cfg
import torch
from torch import optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import time
from custom.utils.logger import Logger
from custom.utils.model_backup import model_backup
from custom.utils.lr_scheduler import WarmupMultiStepLR
from custom.utils.tensorboad_utils import get_writer
from custom.utils.validation_show import save_result_as_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="IA", type=str)
    args = parser.parse_args()
    return args

def train(task):
    if task == "stenosis":
        network_cfg = stenosis_segmentation_cfg
    elif task == "IA":
        network_cfg = IA_segmentation_cfg
    else:
        raise ValueError("Value must be stenosis or IA")

    # 训练准备
    os.makedirs(network_cfg.checkpoints_dir,exist_ok=True)
    logger_dir = network_cfg.log_dir
    os.makedirs(logger_dir, exist_ok=True)
    logger = Logger(logger_dir+"/trainlog.txt", level='debug').logger
    model_backup(logger_dir+"/backup.tar")
    tensorboad_dir = logger_dir + "/tf_logs"
    if os.path.exists(tensorboad_dir): 
        shutil.rmtree(tensorboad_dir)
    writer = get_writer(tensorboad_dir)

    # 网络定义
    SegNet = network_cfg.network.cuda()
    # 定义损失函数
    loss_f = network_cfg.loss_f

    if os.path.exists(network_cfg.load_from):
        print("Load pretrain weight from: " + network_cfg.load_from)
        SegNet.load_state_dict(torch.load(network_cfg.load_from, map_location=network_cfg.device))

    train_dataset = network_cfg.train_dataset
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=network_cfg.batchsize, 
                                shuffle=network_cfg.shuffle,
                                num_workers=network_cfg.num_workers, 
                                drop_last=network_cfg.drop_last)
    valid_dataset = network_cfg.valid_dataset
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=network_cfg.batchsize, 
                                shuffle=False,
                                num_workers=network_cfg.num_workers, 
                                drop_last=network_cfg.drop_last)
    
    trainable_paras = filter(lambda x: x.requires_grad, SegNet.parameters())
    optimizer = optim.AdamW(params=trainable_paras, lr=network_cfg.lr, weight_decay=network_cfg.weight_decay)

    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                milestones=network_cfg.milestones,
                                gamma=network_cfg.gamma,
                                warmup_factor=network_cfg.warmup_factor,
                                warmup_iters=network_cfg.warmup_iters,
                                warmup_method=network_cfg.warmup_method,
                                last_epoch=network_cfg.last_epoch)
    
    
    time_start=time.time()
    for epoch in range(network_cfg.total_epochs): 
        #Training Step!
        SegNet.train()
        for ii, (image, label, vesseldomin) in enumerate(train_dataloader):
            image = V(image).cuda()
            label = V(label).cuda()
            vesseldomin = V(vesseldomin).cuda()
            pred = SegNet(image)
            t_loss = loss_f(pred, label, vesseldomin)
            t_losses = V(torch.zeros(1)).cuda()
            loss_info = ""
            for loss_item, loss_val in t_loss.items():
                t_losses += loss_val
                loss_info += "{}={:.4f}\t ".format(loss_item,loss_val.item())
                writer.add_scalar('TrainDiscLoss/{}'.format(loss_item),loss_val.item(), epoch*len(train_dataloader)+ii+1)
            time_temp=time.time()
            eta = ((network_cfg.total_epochs-epoch)+(1-(ii+1)/len(train_dataloader)))/(epoch+(ii+1)/len(train_dataloader))*(time_temp-time_start)/60
            if eta < 60:
                eta = "{:.1f}min".format(eta)
            else:
                eta = "{:.1f}h".format(eta/60.0)
            logger.info('Epoch:[{}/{}]\t Iter:[{}/{}]\t Eta:{}\t {}'.format(epoch+1 ,network_cfg.total_epochs, ii+1, len(train_dataloader), eta, loss_info))
            optimizer.zero_grad()
            t_losses.backward()
            optimizer.step()

        writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        scheduler.step()

        # Valid Step!
        if (epoch+1) % network_cfg.valid_interval == 0:
            valid_loss = dict()
            SegNet.eval()
            for ii, (image, label, vesseldomin) in enumerate(valid_dataloader):
                image = V(image).cuda()
                label = V(label).cuda()
                vesseldomin = V(vesseldomin).cuda()
                with torch.no_grad():
                    pred = SegNet(image)
                    v_loss = loss_f(pred, label, vesseldomin)             
                    if (ii+1) % 1 == 0: 
                        save_dir = network_cfg.log_dir+"/sample/{}_{}".format(epoch+1, ii+1)
                        os.makedirs(save_dir, exist_ok=True)
                        save_result_as_image(image, pred, label, save_dir)

                for loss_item, loss_val in v_loss.items():
                    if loss_item not in valid_loss:
                        valid_loss[loss_item] = loss_val.item()
                    else:
                        valid_loss[loss_item] += loss_val.item() 
            loss_info = ""              
            for loss_item, loss_val in valid_loss.items():
                valid_loss[loss_item] /= (ii+1)
                loss_info += "{}={:.4f}\t ".format(loss_item,valid_loss[loss_item])
                writer.add_scalar('ValidLoss/{}'.format(loss_item),valid_loss[loss_item], (epoch+1)*len(train_dataloader))
            
            logger.info('Validating Step:\t {}'.format(loss_info))
            
        if (epoch+1) % network_cfg.checkpoint_save_interval == 0:
            torch.save(SegNet.state_dict(), network_cfg.checkpoints_dir+"/{}.pth".format(epoch+1))
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    train(task=args.task)