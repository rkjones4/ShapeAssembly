import torch
import torch.nn as nn
from collections import namedtuple
import etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule

class PCEncoder(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(PCEncoder, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=use_xyz)
        )

        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(256, bn=False, activation=None)
        )

    def forward(self, pointcloud):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz = pointcloud.contiguous()
        features = pointcloud.transpose(1, 2).contiguous()

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))


if __name__ == '__main__':
    device = torch.device('cuda')
    a = torch.randn(10,10000,3).to(device)
    enc = PCEncoder().to(device)
    print(enc(a).shape)
