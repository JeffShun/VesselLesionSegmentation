import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(reduce=False)
        self.smoothing = smoothing
    
    def forward(self, inputs, targets, weights=None):
        if weights is None:
            weights = torch.ones_like(targets)
        assert 0 <= self.smoothing < 1
        targets = targets  * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = weights * self.bceloss(inputs, targets)
        return loss.sum() / weights.sum()
    

class SamplingBCELoss(nn.Module):
    def __init__(self, neg_ratio=10, beta=1, min_sampling=1000):
        super(SamplingBCELoss, self).__init__()
        self.neg_ratio = neg_ratio
        self.beta = beta
        self.min_sampling = min_sampling
        self.bceloss = nn.BCELoss(reduce=False)
    
    def forward(self, inputs, targets, weights=None):
        if weights is None:
            weights = torch.ones_like(targets)
        label_smooth = 0.1
        N = targets.shape[0]
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)
        weights = weights.view(N, -1)
        smooth_targets = targets * (1 - label_smooth * 2) + label_smooth
        loss_all = self.bceloss(inputs, smooth_targets)
        pos_weights = targets

        neg_p = weights * (1-targets)
        loss_neg = loss_all * neg_p
        # softmax_func with weights
        exp_inputs = torch.exp(loss_neg * self.beta)
        exp_inputs = exp_inputs * neg_p
        exp_sum = torch.sum(exp_inputs, 1, keepdim=True)
        loss_neg_normed = exp_inputs / exp_sum

        n_pos = torch.sum(targets, 1, keepdim=True)
        sampling_prob = torch.max(self.neg_ratio * n_pos, torch.zeros_like(n_pos)+self.min_sampling) * loss_neg_normed
        random_map = torch.rand_like(sampling_prob)
        neg_weights = (random_map < sampling_prob).int()
        weights = neg_weights + pos_weights
        # print(torch.sum(neg_weights*neg_p,1)/torch.sum(targets,1))
        loss = (loss_all * weights).sum()/(weights.sum())
        return loss
    

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    
    def forward(self, inputs, targets, weights=None):
        if weights is None:
            weights = torch.ones_like(targets)
        N = targets.size(0)
        smooth = 1e-5
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        weights_flat = weights.view(N, -1)
    
        # 计算交集
        intersection = input_flat * targets_flat * weights_flat 
        N_dice_eff = (2 * intersection.sum(1) + smooth) / ((input_flat*weights_flat).sum(1) + (targets_flat*weights_flat).sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss
    

class Sensitivity_SpecificityLoss(nn.Module):
    def __init__(self, alpha_sen=0.6):
        super(Sensitivity_SpecificityLoss, self).__init__()
        self.alpha_sen = alpha_sen
    
    def forward(self, inputs, targets, weights=None):
        if weights is None:
            weights = torch.ones_like(targets)
        inputs = inputs * weights    
        N = targets.size(0)
        smooth = 1e-5
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        bg_targets_flat = ((1 - targets) * weights).view(N, -1)
        
        sensitivity_loss = ((input_flat - targets_flat)**2 * targets_flat).sum()/(targets_flat.sum()+smooth)
        specificity_loss = ((input_flat - targets_flat)**2 * bg_targets_flat).sum()/(bg_targets_flat.sum()+smooth)
        loss = self.alpha_sen*sensitivity_loss + (1-self.alpha_sen)*specificity_loss
        return loss


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
    
    def forward(self, inputs, weights):
        loss = (inputs * (1 - weights)).mean()    
        return loss
    
class MixLoss(nn.Module):
    def __init__(self, BinaryDiceLoss_weight=1.0, BCELoss_weight=0.0, SBCELoss_weight=0.0, Sensitivity_SpecificityLoss_weight=0.0, MaskLoss_weight=1.0):
        super(MixLoss, self).__init__()
        self.BinaryDiceLoss_weight = BinaryDiceLoss_weight
        self.BCELoss_weight = BCELoss_weight
        self.SBCELoss_weight = SBCELoss_weight
        self.Sensitivity_SpecificityLoss_weight = Sensitivity_SpecificityLoss_weight
        self.MaskLoss_weight = MaskLoss_weight
        self.BinaryDiceLoss = BinaryDiceLoss()
        self.BCELoss = BCELoss(smoothing=0.2)
        self.SBCELoss = SamplingBCELoss(neg_ratio=10, beta=2, min_sampling=1000)
        self.Sensitivity_SpecificityLoss = Sensitivity_SpecificityLoss(alpha_sen=0.5)
        self.MaskLoss = MaskLoss()
    
    def forward(self, inputs, targets, weights):
        loss = dict()
        if self.BinaryDiceLoss_weight > 0:
            BDiceLoss = self.BinaryDiceLoss_weight * self.BinaryDiceLoss(inputs, targets, weights)
            loss.update({"BDiceLoss":BDiceLoss})
        if self.BCELoss_weight > 0:
            BCELoss = self.BCELoss_weight * self.BCELoss(inputs, targets, weights)
            loss.update({"BCELoss":BCELoss})
        if self.SBCELoss_weight > 0:
            SBCELoss = self.SBCELoss_weight * self.SBCELoss(inputs, targets, weights)
            loss.update({"SBCELoss":SBCELoss})
        if self.Sensitivity_SpecificityLoss_weight > 0:
            SSLoss = self.Sensitivity_SpecificityLoss_weight * self.Sensitivity_SpecificityLoss(inputs, targets, weights)
            loss.update({"SSLoss":SSLoss})
        if self.MaskLoss_weight > 0:
            MaskLoss = self.MaskLoss_weight * self.MaskLoss(inputs, weights)
            loss.update({"MaskLoss":MaskLoss})
        return loss
    

