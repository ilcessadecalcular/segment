import torch
import torch._utils
import torch.nn as nn
from models.twoD_rnn.bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
import numpy as np
import torch.nn.functional as F
from models.twoD_rnn.config import HRNet48, HRNet32
import os




BN_MOMENTUM = 0.1
ALIGN_CORNERS = None

# logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out



class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            # logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            # logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            # logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)
                        ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)
                                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNetSeg(nn.Module):
    def __init__(self,config, **kwargs):
        global ALIGN_CORNERS
        super(HRNetSeg, self).__init__()
        # 我需要定义的层
        ALIGN_CORNERS = None
        HRNetSeg_config = config()
        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = HRNetSeg_config.STAGE1()
        num_channels = self.stage1_cfg.NUM_CHANNELS[0]
        block = blocks_dict[self.stage1_cfg.BLOCK]
        num_blocks = self.stage1_cfg.NUM_BLOCKS[0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = HRNetSeg_config.STAGE2()
        num_channels = self.stage2_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage2_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = HRNetSeg_config.STAGE3()
        num_channels = self.stage3_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage3_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = HRNetSeg_config.STAGE4()
        num_channels = self.stage4_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage4_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=HRNetSeg_config.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if HRNetSeg_config.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)
                        ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config.NUM_MODULES
        num_branches = layer_config.NUM_BRANCHES
        num_blocks = layer_config.NUM_BLOCKS
        num_channels = layer_config.NUM_CHANNELS
        block = blocks_dict[layer_config.BLOCK]
        fuse_method = layer_config.FUSE_METHOD

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg.NUM_BRANCHES:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg.NUM_BRANCHES:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x
        

class RNNSeg(nn.Module):
    def __init__(self,config, num_feat=18, num_blocks=10):
        super(RNNSeg, self).__init__()
        self.num_feat=num_feat
        self.config = config()
        self.hrnet_seg=HRNetSeg(config)
        self.num_blocks = num_blocks
        self.backward_resblocks = ResidualBlocksWithInputConv(
            num_feat*2, num_feat, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            num_feat*2, num_feat, num_blocks)
        self.fusion = nn.Conv2d(
            num_feat * 2, num_feat, 1, 1, 0, bias=True)
        #self.ln1 = nn.LayerNorm([num_feat,64,64],elementwise_affine=False) 
        self.up = nn.Sequential(
            nn.ConvTranspose2d(num_feat,num_feat,kernel_size=2,stride=2),
            nn.Conv2d(num_feat,num_feat,3,1,1),
            #nn.LayerNorm([num_feat,128,128]),
            #BatchNorm2d(num_feat,momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(num_feat, num_feat, kernel_size=2, stride=2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            #BatchNorm2d(num_feat, momentum=BN_MOMENTUM),
            #nn.LayerNorm([num_feat,256,256]),
            nn.ReLU(inplace=False))
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #self.ln2 = nn.LayerNorm([num_feat,256,256],elementwise_affine=False)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #self.ln3 = nn.LayerNorm([num_feat,256,256],elementwise_affine=False)
        self.last = nn.Conv2d(num_feat, 1, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)
        #self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        # x是一系列切片 x:b,c,t,h,w
        x = torch.permute(x,(0,2,1,3,4))
        #x:b,t,c,h,w
        input = x[0,:,:,:,:]
        mid_out = self.hrnet_seg(input)
        mid_out = torch.permute(mid_out,(1,0,2,3))
        mid_out = mid_out.unsqueeze(0)
        #print(mid_out.shape)
        #这里就把维度变回来  b,c,t,h,w
        b, c, t, h, w = mid_out.shape
        outputs = []
        feat_prop = mid_out.new_zeros(b, c, h, w)
        #output = torch.ones(b, c, h, w)
        for i in range(t - 1, -1, -1):
            #if i < t - 1:  # no warping required for the last timestep
                #feat_prop = mid_out[:, :, i+1, :, :]
                #out:b,c,t,h,w
                #feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([mid_out[:, :, i, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            #print(mid_out.shape)
            output_curr = mid_out[:, :, i, :, :]
            # out:b,c,t,h,w
            #if i > 0:  # no warping required for the first timestep
            #     if flows_forward is not None:
            #         flow = flows_forward[:, i - 1, :, :, :]
            #     else:
            #         flow = flows_backward[:, -i, :, :, :]
                 #feat_prop = mid_out[:, :, i-1, :, :]

            #forward backward fenkai
            feat_prop = torch.cat([output_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.relu(self.fusion(out))

            # feat_prop = torch.cat([output_curr, feat_prop, outputs[i]], dim=1)
            # feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features

            #out = self.fusion(out)
            #out = self.ln1(out)
            #out = self.relu(out)
            
            
            out = self.up(out)
            out = self.relu(self.conv1(out))
            out = self.relu(self.conv2(out))
            
            #out = self.conv1(out)
            #out = self.ln2(out)
            #out = self.relu(out)
            #out = self.conv2(out)
            #out = self.ln3(out)
            #out = self.relu(out)
            
            
            out = self.last(out)
            # base = self.img_upsample(lr_curr)
            #value,index = torch.max(out,dim=1)
            #print(value)
            # out += base
            outputs[i] = out

        return torch.stack(outputs, dim=2)


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=32, num_blocks=10):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        #main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        #main.append(nn.LayerNorm([out_channels,64,64],elementwise_affine=False))
        #main.append(nn.ReLU(inplace=True))
        #main.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))
        #main.append(nn.LayerNorm([out_channels,64,64],elementwise_affine=False))
        
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=32, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.ln1 = nn.LayerNorm([mid_channels,64,64],elementwise_affine=False)
        self.ln2 = nn.LayerNorm([mid_channels, 64, 64], elementwise_affine=False)
        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.

        #if res_scale == 1.0:
        #    self.init_weights()

    # def init_weights(self):
    #     """Initialize weights for ResidualBlockNoBN.
    #
    #     Initialization methods like `kaiming_init` are for VGG-style
    #     modules. For modules with residual paths, using smaller std is
    #     better for stability and performance. We empirically use 0.1.
    #     See more details in "ESRGAN: Enhanced Super-Resolution Generative
    #     Adversarial Networks"
    #     """
    #
    #     for m in [self.conv1, self.conv2]:
    #         default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        #out = self.conv2(self.relu(self.conv1(x)))

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)

        return identity + out * self.res_scale

# def default_init_weights(module, scale=1):
#     """Initialize network weights.
#
#     Args:
#         modules (nn.Module): Modules to be initialized.
#         scale (float): Scale initialized weights, especially for residual
#             blocks.
#     """
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             kaiming_init(m, a=0, mode='fan_in', bias=0)
#             m.weight.data *= scale
#         elif isinstance(m, nn.Linear):
#             kaiming_init(m, a=0, mode='fan_in', bias=0)
#             m.weight.data *= scale
#         elif isinstance(m, _BatchNorm):
#             constant_init(m.weight, val=1, bias=0)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)



def main():
    #x=torch.ones([1,1,60,256,256],dtype=torch.float32).cuda()
    x = torch.ones(1, 1, 60, 256, 256)
    #model = HRNetSeg(HRNet48).to('cuda:0')
    # only_twod=OnlyHRNetSeg(HRNet48).to('cuda:0')
    # output=only_twod(x)
    #output=model(x)
    #####

    RNNSeggg=RNNSeg(HRNet32).cuda()
    label=RNNSeggg(x)
    print(label.shape)
if __name__ == '__main__':
    main()
