import torch
import torch.nn as nn

class Segmentation_Network(nn.Module):
    def __init__(
        self, 
        backbone,
        channels, 
        n_class,
        ):
        super(Segmentation_Network, self).__init__()
        self.backbone = backbone
        self.out = nn.Sequential(
            nn.Conv3d(channels, n_class, 1),
            nn.Sigmoid()
            )
        
        self.initialize_weights()

    @torch.jit.export
    def forward(self, img):
        out = self.backbone(img)
        out = self.out(out)
        return out


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.neck = nn.SyncBatchNorm.convert_sync_batchnorm(self.neck)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
