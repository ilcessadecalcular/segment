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
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from utils.distributed_utils import init_distributed_mode, dist, cleanup, is_main_process, reduce_value
from utils.train_eval_utils import train_one_epoch,evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import random
from thop import profile
from ptflops import get_model_complexity_info



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

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint, help='Number of epochs per checkpoint')
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
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    return parser

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    ###  improvement
    torch.backends.cudnn.enabled = True


def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    seed_torch()

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = args.cudnn_enabled
    # torch.backends.cudnn.benchmark = args.cudnn_benchmark


    from data_function import MedData_train,MedData_eval
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=logs", view at http://localhost:6006/')
        writer = SummaryWriter(args.output_dir)


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

    from models.twoD_rnn.OnlyHRNet import get_seg_model
    from models.twoD_rnn.config import HRNet8
    model = get_seg_model(HRNet8,in_feat=HRNet8.DATASET.NUM_CLASSES).to(device)

    # from models.twoD_rnn.RNN_HRNet2 import RNNSeg
    # from models.twoD_rnn.config import HRNet8
    # model = RNNSeg(HRNet8,num_feat=HRNet8.DATASET.NUM_CLASSES).to(device)

    # from models.twoD_rnn.HRNet_doubleRNN import RNNSeg
    # from models.twoD_rnn.config import HRNet8
    # model = RNNSeg(HRNet8).to(device)

    if rank == 0:
        flops, params = get_model_complexity_info(model,(1,hp.crop_size,256,256))
        print('Flops:' + flops)
        print('Params:' + params)

    # flops, params = profile(model, inputs=(40,256,256))
    # print(flops,params)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)


    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    #scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 50, eta_min=0.01*args.init_lr,T_mult=2)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        elapsed_epochs = 0

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])



    from loss_function import Binary_Loss,DiceLoss,BinaryDiceLoss,BCEFocalLoss
    #criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion = Binary_Loss().to(device)
    # criterion = BinaryDiceLoss().to(device)
    #criterion = BCEFocalLoss().to(device)
    


    train_dataset = MedData_train(source_train_dir,label_train_dir)
    eval_dataset = MedData_eval(source_eval_dir,label_eval_dir)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch, drop_last=True)

    # nw = min([os.cpu_count(), args.batch if args.batch > 1 else 0, 8])  # number of workers
    # if rank == 0:
    #     print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset.training_set,
                            # num_workers=nw,
                            pin_memory=True,
                            batch_sampler=train_batch_sampler
                            )

    eval_loader = DataLoader(eval_dataset.training_set,
                            batch_size=1,
                            sampler=eval_sampler,
                            pin_memory=True,
                            # num_workers=nw
                            )

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        train_sampler.set_epoch(epoch)
        epoch += elapsed_epochs
        num_iters = 0

        # # 在进程0中打印训练进度
        # if is_main_process():
        #     train_loader = tqdm(train_loader, file=sys.stdout)

        mean_loss, train_dice = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    criterion=criterion,
                                    crop_size=hp.crop_size
                                    )
        scheduler.step()

        eval_dice = evaluate(model=model,
                        data_loader=eval_loader,
                        device=device,
                        epoch=epoch
                        )
        torch.cuda.empty_cache()
            ## log
        if rank == 0:
            print("mean_loss:" + str(mean_loss))
            print('lr:' + str(scheduler._last_lr[0]))
            print('eval_dice' + str(eval_dice))
            writer.add_scalar('Training/Loss', mean_loss,epoch)
            writer.add_scalar('Training/dice', train_dice,epoch)
            # writer.add_scalar('Training/hd95', train_hd95, epoch)
            # writer.add_scalar('Training/precision', train_precision, epoch)
            # writer.add_scalar('Training/recall', train_recall, epoch)
            # writer.add_scalar('Training/asd', train_asd, epoch)
            #writer.add_scalar('Eval/false_positive_rate', false_positive_rate,epoch)
            #writer.add_scalar('Eval/false_negtive_rate', false_negtive_rate,epoch)
            writer.add_scalar('Eval/dice', eval_dice,epoch)
            # writer.add_scalar('Eval/hd95', eval_hd95, epoch)
            # writer.add_scalar('Eval/precision', eval_precision, epoch)
            # writer.add_scalar('Eval/recall', eval_recall, epoch)
            # writer.add_scalar('Eval/asd', eval_asd, epoch)


        # Store latest checkpoint in each epoch
        if rank == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "epoch": epoch,

                },
                os.path.join(args.output_dir, args.latest_checkpoint_file),
            )




        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            if rank == 0:
                torch.save(
                    {

                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
                )




                # with torch.no_grad():
                #
                #
                #     x = x[0].cpu().detach().numpy()
                #     y = y[0].cpu().detach().numpy()
                #     outputs = outputs[0].cpu().detach().numpy()
                #     affine = batch['source']['affine'][0].numpy()
                #
                #
                #
                #
                #     source_image = torchio.ScalarImage(tensor=x, affine=affine)
                #     source_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-source"+hp.save_arch))
                #     # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))
                #
                #     label_image = torchio.ScalarImage(tensor=y, affine=affine)
                #     label_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt"+hp.save_arch))
                #
                #     output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
                #     output_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict"+hp.save_arch))

    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    writer.close()
    cleanup()

def al_test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_dir_test, exist_ok=True)


    #from models.three_d.unet3d import UNet3D
    #model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32)

    from models.three_d.residual_unet3d import UNet
    model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=2)

    #from models.three_d.fcn3d import FCN_Net
    #model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

    #from models.three_d.highresnet import HighRes3DNet
    #model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

    #from models.three_d.densenet3d import SkipDenseNet3D
    #model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

    # from models.three_d.densevoxelnet3d import DenseVoxelNet
    # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

    #from models.three_d.vnet3d import VNet
    #model = VNet(in_channels=hp.in_class, classes=hp.out_class)

    #from models.three_d.unetr import UNETR
    #model = UNETR(img_shape=(hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)


    # model = torch.nn.DataParallel(model, device_ids=devicess)


    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    model.cuda()



    test_dataset = MedData_test(source_test_dir,label_test_dir)
    znorm = ZNormalization()

    patch_overlap = hp.patch_overlap
    patch_size = hp.patch_size



    for i,subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
                subj,
                patch_size,
                patch_overlap,
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=args.batch)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):


                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]


                outputs = model(input_tensor)


                logits = torch.sigmoid(outputs)

                labels = logits.clone()
                labels[labels>0.5] = 1
                labels[labels<=0.5] = 0

                aggregator.add_batch(logits, locations)
                aggregator_1.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()




        affine = subj['source']['affine']


        label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
        label_image.save(os.path.join(output_dir_test,f"{i:04d}-result_float"+hp.save_arch))

        # f"{str(i):04d}-result_float.mhd"

        output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
        output_image.save(os.path.join(output_dir_test,f"{i:04d}-result_int"+hp.save_arch))


   

if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        al_test()
