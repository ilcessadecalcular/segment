import os
import sys
import tempfile
import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.distributed_utils import init_distributed_mode, dist, cleanup, is_main_process, reduce_value
from utils.train_eval_utils import train_one_epoch, evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import random
from thop import profile
from ptflops import get_model_complexity_info
from utils.Testutils import p_resample
from torch.nn import functional as F
import SimpleITK as sitk

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir

source_eval_dir = hp.source_eval_dir
label_eval_dir = hp.label_eval_dir

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_dir_test = hp.output_dir_test


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--model_test_dir', type=str, default=hp.model_test_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    training.add_argument('--divide', type=int, default=hp.divide, help='divide-size')
    # training.add_argument('--crop_size', type=int, default=hp.crop_size, help='crop-size')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument(
        "--rank", type=int, default=-1, help="rank for distributed training")

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    return parser


def test_seg():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import  MedData_test
    os.makedirs(output_dir_test, exist_ok=True)

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)


    # from models.three_d.unet3d import UNet3D
    # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32).to(device)

    # from models.three_d.residual_unet3d import UNet
    # model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=2).to(device)

    # from models.three_d.fcn3d import FCN_Net
    # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class).to(device)

    # from models.three_d.highresnet import HighRes3DNet
    # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class).to(device)

    # from models.three_d.densenet3d import SkipDenseNet3D
    # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class).to(device)

    # from models.three_d.densevoxelnet3d import DenseVoxelNet
    # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class).to(device)

    # from models.three_d.vnet3d import VNet
    # model = VNet(in_channels=hp.in_class, classes=hp.out_class).to(device)

    # from models.three_d.unetr import UNETR
    # model = UNETR(img_shape=(hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class).to(device)

    # from models.twoD.unet import only_unet
    # model = only_unet(in_channels=hp.in_class, classes=hp.out_class).to(device)

    # from models.three_d.hrnet3d import Hrnet_3d
    # from models.twoD_rnn.config import HRNet48
    # model = Hrnet_3d(HRNet48).to(device)

    # from models.twoD_rnn.OnlyHRNet import get_seg_model
    # from models.twoD_rnn.config import HRNet8
    # model = get_seg_model(HRNet8,in_feat=HRNet8.DATASET.NUM_CLASSES).to(device)

    from models.twoD_rnn.RNN_HRNet2 import RNNSeg
    from models.twoD_rnn.config import HRNet8
    model = RNNSeg(HRNet8, num_feat=HRNet8.DATASET.NUM_CLASSES).to(device)

    # from models.twoD_rnn.HRNet_doubleRNN import RNNSeg
    # from models.twoD_rnn.config import HRNet8
    # model = RNNSeg(HRNet8).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    # print("load model:", args.ckpt)
    print(os.path.join(args.model_test_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.model_test_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])

    test_dataset = MedData_test(source_test_dir)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             pin_memory=True,
                             # num_workers=nw
                             )
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            patient,img,spacing,origin,direction,x,y,h,w = batch
            p_patient = patient[0]
            print(p_patient)
            print(img.shape)
            img1 = img[:,:,:300,:,:].float()
            out1 = model(img1)
            img2 = img[:,:,300:,:,:].float()
            out2 = model(img2)
            out = torch.cat((out1,out2),2)
            label = torch.sigmoid(out)
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
            label = label[0,0,:,:,:]
            print(label.shape)

            pad_length_x_left = x - 128
            pad_length_x_right = w - x - 128
            pad_length_y_front = y - 128
            pad_length_y_back = h - y - 128
            padding = (pad_length_x_left, pad_length_x_right,
                       pad_length_y_front, pad_length_y_back)
            p_label = F.pad(label,padding)
            print(p_label.shape)

            resample_spacing = np.array(spacing, dtype=np.float32)
            resample_spacing = resample_spacing[::-1]
            model_spacing = [1.0, 0.7, 0.7]
            p_label = p_resample(p_label,model_spacing,resample_spacing)
            print(p_label.shape)
            image3 = sitk.GetImageFromArray(p_label)
            n_spacing = [float(x) for x in spacing]
            n_direction = [float(x) for x in direction]
            n_origin = [float(x) for x in origin]
            image3.SetSpacing(n_spacing)
            image3.SetDirection(n_direction)
            image3.SetOrigin(n_origin)
            nii_out_path = os.path.join(output_dir_test, p_patient + '.nii.gz')
            sitk.WriteImage(image3, nii_out_path)


if __name__ == '__main__':
    test_seg()