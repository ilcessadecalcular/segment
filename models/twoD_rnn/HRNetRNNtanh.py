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
        self.tanh = nn.Tanh()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.tanh(out)

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
        self.tanh = nn.Tanh()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.tanh(out)

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
        self.tanh = nn.Tanh()

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
                                nn.Tanh()))
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
            x_fuse.append(self.tanh(y))

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
        self.tanh = nn.Tanh()

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
            nn.Tanh(),
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
                        nn.Tanh()))
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
                        nn.Tanh()
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
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh(x)
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
        

    #
    #
    #
    #
    # def forward(self,x):
    #     # 2d分割逻辑
    #
    #     # x是一张切片
    #     return x


# class RNNSeg(nn.Module):
#     def __init__(self,num_feat):
#         super(RNNSeg, self).__init__()
#         self.num_feat=num_feat
#         self.hrnet_seg=HRNetSeg(num_feat,)
#         self.up = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
#             nn.PixelShuffle(2),
#             nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
#             nn.PixelShuffle(2)
#         )
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.last = nn.Conv2d(num_feat, out_feat, 3, 1, 1)
#         self.softmax = nn.Softmax(dim=1)
#     def forward(self,x):
#         # x是一系列切片
#         b,n,_,h,w=x.shape
#         # x:[b,n,1,h,w]
#         # 假定2d分割的特征通道数为d
#         hidden_state=x.zeros_like(b,self.num_feat,h,w)
#         outputs=[]
#         for i in range(n):
#             out=torch.cat([hidden_state,x[:,i,:,:,:]],dim=1)
#             out=self.hrnet_seg(out)
#             outputs.append(out)
#             hidden_state=out
#         real_outputs=[]
#         for out in outputs:
#             real_out=self.up(out)
#             real_out = self.conv(self.relu(real_out))
#             real_out = self.last(self.relu(real_out))
#             real_out = self.softmax(real_out)
#             real_outputs.append(real_out)
#         pass
#
class OnlyHRNetSeg(nn.Module):
    def __init__(self, config,in_feat=1, mid_feat=1, out_feat=1):
        super(OnlyHRNetSeg, self).__init__()
        # self.down=nn.Conv2d(in_feat,mid_feat,3,stride=4,padding=1)
        self.config = config()
        self.hrnet_seg=HRNetSeg(config)
        self.up=nn.Sequential(
            nn.Conv2d(mid_feat, mid_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(mid_feat, mid_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        #self.tanh=nn.Tanh()
        #self.sigmoid=nn.Sigmoid()
        #self.relu=nn.ReLU(inplace=True)
        #self.conv=nn.Conv2d(mid_feat, mid_feat, 3, 1, 1)
        #self.last=nn.Conv2d(mid_feat, out_feat, 3, 1, 1)
        #self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        b,_,d,h,w=x.shape
        outputs=[]
        for i in range(d):
            out=x[:,:,i,:,:]
            # out=self.down(x[:,i,:,:,:])
            out=self.hrnet_seg(out)
            outputs.append(out)
        real_outputs=[]
        for out in outputs:
            real_out=self.up(out)
            #real_out=self.tanh(real_out)
            #real_out=self.sigmoid(real_out)
            #real_out=self.conv(self.relu(real_out))
            #real_out=self.last(self.relu(real_out))
            #real_out=self.softmax(real_out)
            real_outputs.append(real_out)
        real_outputs=torch.stack(real_outputs).permute(1,2,0,3,4)
        return real_outputs

    def init_weights(self, pretrained='',):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            
def get_seg_model(cfg, **kwargs):
    model = OnlyHRNetSeg(cfg, **kwargs)
    #model.init_weights(cfg.PRETRAINED)

    return model


def main():
    x=torch.ones([1,1,1,256,256],dtype=torch.float32).cuda()
    #only_twod=OnlyHRNetSeg(HRNet48).to('cuda:0')
    #output=only_twod(x)
    model=HRNetSeg(HRNet48).to('cuda:0')
    output=model(x)
    print(output.shape)
if __name__ == '__main__':
    main()
