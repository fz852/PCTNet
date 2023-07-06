import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils import SpatialTransformer, UnetrUpBlock1, UnetrUpBlock2
from typing import Tuple
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from typing import Union
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from pctnet_encoder import pctnet_conv
from torch.nn import  Softmax



def INF3DH(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W * D, 1, 1)  # .cuda()


def INF3DW(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(W), 0).unsqueeze(0).repeat(B * H * D, 1, 1)  # .cuda()


def INF3DD(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(D), 0).unsqueeze(0).repeat(B * H * W, 1, 1)  # .cuda()

class SIA(nn.Module):


    def __init__(self, in_dim, verbose=False):
        super(SIA, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=4)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.verbose = verbose
        self.INFH = INF3DH
        self.INFD = INF3DD

    # def forward(self, proj_query,proj_key,proj_value):
    def forward(self, x):
        m_batchsize, _, height, width, depth = x.size()  # proj_query.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1,
                                                                           height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1,
                                                                           width).permute(0, 2, 1)
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1,
                                                                           depth).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1, height)
        proj_key_W = proj_key.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1, width)
        proj_key_D = proj_key.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1, depth)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width * depth, -1, height)
        proj_value_W = proj_value.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height * depth, -1, width)
        proj_value_D = proj_value.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1, depth)


        inf_holder = self.INFH(m_batchsize, height, width, depth).to(x.device)
        energy_H = torch.bmm(proj_query_H, proj_key_H) + inf_holder
        energy_H = energy_H.view(m_batchsize, width, depth, height, height).permute(0, 1, 3, 2, 4)


        energy_W = torch.bmm(proj_query_W, proj_key_W)
        energy_W = energy_W.view(m_batchsize, height, depth, width, width).permute(0, 3, 1, 2, 4)
        energy_D = (torch.bmm(proj_query_D, proj_key_D) + self.INFD(m_batchsize, height, width, depth).to(
            x.device)).view(m_batchsize, height, width, depth, depth).permute(0, 2, 1, 3, 4)  # bwhdd

        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4))
        att_H = concate[:, :, :, :, 0:height].permute(0, 1, 3, 2, 4).contiguous().view(m_batchsize * width * depth,
                                                                                       height, height)
        att_W = concate[:, :, :, :, height:height + width].permute(0, 2, 3, 1, 4).contiguous().view(
            m_batchsize * height * depth, width, width)
        att_D = concate[:, :, :, :, height + width:].permute(0, 2, 1, 3, 4).contiguous().view(
            m_batchsize * height * width, depth, depth)


        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, depth, -1, height).permute(0,
                                                                                                                    3,
                                                                                                                    4,
                                                                                                                    1,
                                                                                                                    2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, depth, -1, width).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    4,
                                                                                                                    2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize, height, width, -1, depth).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    4)

        return self.gamma * (out_H + out_W + out_D) + x

class RSIA(nn.Module):
    def __init__(self, in_channels):
        super(RSIA, self).__init__()
        inter_channels = in_channels
        self.sia = SIA(inter_channels)


    def forward(self, x, recurrence=2):
        for i in range(recurrence):
            output = self.sia(x)
        return output


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x




class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class PCTNet(nn.Module):

    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            channel_dim=[768, 384, 192, 96],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.channel_dim = channel_dim
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims
        self.is_train = True

        self.spatial_trans = SpatialTransformer([160, 192, 224])
        self.pctnet_3d = pctnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=48,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock1(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock1(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock1(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock1(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


        self.decoder1 = UnetrUpBlock2(
            spatial_dims=spatial_dims,
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        self.sia0 = RSIA(96)
        self.sia1 = RSIA(192)
        self.sia2 = RSIA(384)
        self.sia3 = RSIA(768)
        # self.senet0 = SEModule(96, 6)
        # self.senet1 = SEModule(192, 6)
        # self.senet2 = SEModule(384, 6)
        # self.senet3 = SEModule(768, 6)



        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        # x_in = torch.cat((x, y), dim=1) # x:moving, y:fixed
        outs = self.pctnet_3d(x_in)
        source = x_in[:, 0:1, :, :, :]
        fixed_img = x_in[:, 1:2, :, :, :]

        x_s0 = x_in.clone()
        x_s1 = self.avg_pool(x_in)
        f4 = self.encoder0(x_s1)
        f5 = self.encoder1(x_s0)
        x2 = outs[0]
        f3 = self.encoder2(x2)
        f3 = self.sia0(f3)
        x3 = outs[1]
        f2 = self.encoder3(x3)
        f2 = self.sia1(f2)
        x4 = outs[2]
        f1 = self.encoder4(x4)
        f1 = self.sia2(f1)
        enc_hidden = self.encoder5(outs[3])
        enc_hidden = self.sia3(enc_hidden)
        dec4 = self.decoder5(enc_hidden, f1)
        dec3 = self.decoder4(dec4, f2)
        dec2 = self.decoder3(dec3, f3)
        dec1 = self.decoder2(dec2, f4)
        dec0 = self.decoder1(dec1, f5)
        flow = self.reg_head(dec0)
        out = self.spatial_trans(source, flow)


        return out, flow



