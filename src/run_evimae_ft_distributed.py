# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import json
import torch
import torch.distributed
from torch.utils.data import WeightedRandomSampler
import dataloader as dataloader
import models, random
import numpy as np
import warnings
from traintest_ft import train, validate
from sklearn import metrics
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)


###########################################################


def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataset
    parser.add_argument("--data-train", type=str, default='', help="training data json")
    parser.add_argument("--data-val", type=str, default='', help="validation data json")
    parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
    parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
    parser.add_argument("--n_class", type=int, default=527, help="number of classes")
    parser.add_argument("--model", type=str, default='evi-mae-ft', help="the model used")
    parser.add_argument("--dataset", type=str, default="cmummac", help="the dataset used", choices=["cmummac", "wear"])
    parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

    # training
    parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
    # not used in the formal experiments, only in preliminary experiments
    parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
    parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
    parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
    parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
    parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
    parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)

    parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
    parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
    parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")
    parser.add_argument("--wa_num", type=int, default=12, help="how many epochs to average in finetuning")

    parser.add_argument("--only_val", help='if only do evaluation', type=ast.literal_eval, default='False')

    parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
    parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

    parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
    parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
    parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
    parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

    parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights")
    parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
    parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

    parser.add_argument("--base_lr", type=float, default=1, help="the base learning rate of the model")
    parser.add_argument("--image_as_video", help='if use image as video', type=ast.literal_eval, default='False')

    # imu
    parser.add_argument("--imu_target_length", type=int, default=48, help="the target length of imu data")
    # parser.add_argument("--imu_masking_ratio", type=float, default=0.75, help="masking ratio")
    # parser.add_argument("--imu_mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])
    parser.add_argument("--imu_plot_type", type=str, default='fbank', help="the plot type of imu data", choices=['fbank', 'rp', 'mel', 'raw', 'stft']) 
    parser.add_argument("--imu_plot_height", type=int, default=64, help="the plot height of imu data")
    parser.add_argument("--imu_patch_size", type=int, default=8, help="the patch size of imu data")
    parser.add_argument("--imu_dataset_mean", type=float, help="the dataset imu mean, used for input normalization")
    parser.add_argument("--imu_dataset_std", type=float, help="the dataset imu std, used for input normalization")
    parser.add_argument("--imu_channel_num", type=int, default=12, help="the channel number of imu data")
    parser.add_argument("--imu_encoder_embed_dim", type=int, default=384, help="the embed dim of imu encoder")
    parser.add_argument("--imu_encoder_depth", type=int, default=12, help="the depth of imu encoder")
    parser.add_argument("--imu_encoder_num_heads", type=int, default=6, help="the num heads of imu encoder")
    parser.add_argument("--imu_enable_graph", type=ast.literal_eval, default='False', help="enable graph for imu data")
    parser.add_argument("--imu_graph_net", type=str, default='gin', help="the graph net for imu data", choices=['gin']) # dont use gat
    parser.add_argument("--imu_two_stream", type=ast.literal_eval, default='False', help="if use two mlp")

    # video
    parser.add_argument("--video_img_size", type=int, default=224, help="the image size of video data")
    parser.add_argument("--video_patch_size", type=int, default=16, help="the patch size of video data")
    parser.add_argument("--video_encoder_num_classes", type=int, default=0, help="the number of classes of video encoder")
    parser.add_argument("--video_decoder_num_classes", type=int, default=1536, help="the number of classes of video decoder")
    parser.add_argument("--video_mlp_ratio", type=int, default=4, help="the mlp ratio of video data")
    parser.add_argument("--video_qkv_bias", type=ast.literal_eval, default='True', help="the qkv bias of video data")
    parser.add_argument("--video_encoder_embed_dim", type=int, default=384, help="the embed dim of video encoder")
    parser.add_argument("--video_encoder_depth", type=int, default=12, help="the depth of video encoder")
    parser.add_argument("--video_encoder_num_heads", type=int, default=6, help="the num heads of video encoder")
    parser.add_argument("--video_decoder_embed_dim", type=int, default=192, help="the embed dim of video decoder")
    parser.add_argument("--video_decoder_num_heads", type=int, default=3, help="the num heads of video decoder")
    parser.add_argument("--video_masking_ratio", type=float, default=0.75, help="video masking ratio")

    parser.add_argument("--rseed", type=int, default=42, help="random seed")

    # distributed
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--root', type=str, default='./cifar')

    # wandb
    parser.add_argument("--use_wandb", type=ast.literal_eval, default='True', help="Use wandb logging or not")
    parser.add_argument("--wandb_project", type=str, default='MyProject', help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/team name")

    return parser


###########################################################


def init_for_distributed(rank, args):

    # 1. setting for distributed training
    # 이 부분에서 rank와 local_gpu_id가 매핑됨
    # rank는 각 프로세스를 의미하고, local_gpu_id는 각 프로세스가 사용할 GPU의 ID 라고 보면 됨
    # 그리고 rank 0은 "마스터 프로세스", rank 1, 2, 3은 "워커 프로세스" 라고 함
    # opts.gpu_ids는 [0, 1, 2, 3]
    # 결국 rank 0 -> local_gpu_id 0, .... rank 3 -> local_gpu_id 3 까지 1:1로 매핑됨
    # 그 후, 해당 GPU가 현재 프로세스의 기본 디바이스로 설정됨
    args.rank = rank
    local_gpu_id = int(args.gpu_ids[args.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if args.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.world_size,
                            rank=args.rank)

    # if put this function, the all processes block at all.
    # 모든 프로세스가 이 지점에 도달할 때까지 대기, 프로세스들의 동기화를 보장
    dist.barrier()

    # convert print fn iif rank is zero
    # rank가 0일 때에만 print가 이루어짐!! 여러 프로세스가 동시에 print문을 실행하면 콘솔이 복잡해지니까...
    # 즉, print문 앞에 if opts.rank == 0: 과 같은 조건을 굳이 붙이지 않아도 된다!
    setup_for_distributed(args.rank == 0)
    print(args)
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


###########################################################


# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
# 평균 가중치 모델(weight averaging model)을 생성하고 평가하는 데 사용
# Fine-tuning 도중 또는 완료 후, 여러 체크포인트의 가중치를 평균화하여 모델의 최종 가중치를 재조정하는 과정
# 아래 코드는 Fine-tuning 완료 후에 사용을 하고 있음

# 왜 Pre-training 과정에서는 평균 가중치를 사용하지 않는가??
# Pre-training은 모델이 다양한 데이터에서 **일반적인 표현(representation)**을 학습하도록 하는 과정
# 일반적으로 데이터셋이 크고, 학습 과정이 길기 때문에 특정 에포크에서의 노이즈가 전체 학습 목표에 큰 영향을 미치지 않음
# 오히려 평균화된 가중치를 사용하면 모델이 다양한 패턴을 학습하는 데 방해가 될 수 있음

# 하지만 Fine-tuning은 데이터셋이 더 작고 학습 과정이 짧기 때문에 과적합(overfitting) 위험이 크며, 에포크 간 가중치 변동이 상대적으로 작음
# 평균 가중치를 사용하면 과적합을 방지하고, 학습이 불안정하거나 특정 에포크에서 발생한 노이즈의 영향을 완화할 수 있음

def wa_model(exp_dir, start_epoch, end_epoch, wa_num=12):
    
    # start_epoch에 해당하는 체크포인트를 로드하여 초기 가중치로 설정
    sdA = torch.load(exp_dir + '/models/evi_model.' + str(start_epoch) + '.pth', map_location='cpu')
    
    # 몇 개의 모델이 평균화에 포함되었는지 카운트
    model_cnt = 1

    # choose wa_num models from start_epoch to end_epoch
    # start_epoch에서 end_epoch까지, wa_num 개의 에포크를 균등하게 선택
    # 각 에포크의 가중치를 로드한 후 sdA에 누적
    epoch_list = np.linspace(start_epoch, end_epoch, wa_num, dtype=int)
    for epoch in epoch_list: # range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/evi_model.' + str(epoch) + '.pth', map_location='cpu')
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))

    # 모든 키(모델 파라미터)에 대해 누적된 값을 평균화
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
        
    return sdA



###########################################################


def main_worker(rank, args, imu_conf, val_imu_conf):
    local_gpu_id = init_for_distributed(rank, args)

    # 1. wandb init
    # rank==0 이면 wandb를 초기화, 아니면 쓸모없는 모드('disabled')로 둠
    if args.use_wandb:
        if rank == 0:
            wandb.init(
                project=args.wandb_project, 
                entity=args.wandb_entity,  # entity가 없다면 생략 가능
                name=args.exp_dir,        # run name
                config=vars(args)         # args 전체를 config로 저장
            )
        else:
            # rank 0 이외에는 log를 남기지 않도록 비활성화
            wandb.init(mode="disabled")

    train_set = dataloader.EVIDataset(args.data_train, label_csv=args.label_csv, imu_conf=imu_conf, video_masking_ratio=args.video_masking_ratio, image_as_video=args.image_as_video)
    val_set = dataloader.EVIDataset(args.data_val, label_csv=args.label_csv, imu_conf=val_imu_conf)
    
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    val_sampler = DistributedSampler(dataset=val_set, shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batch_size,
                                               shuffle=False, 
                                               num_workers=args.num_workers,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                               batch_size=args.batch_size,
                                               shuffle=False, 
                                               num_workers=args.num_workers,
                                               sampler=val_sampler,
                                               pin_memory=True,
                                               drop_last=True)

    if args.model == 'evi-mae-ft':
        video_model_dict = {
            'img_size': args.video_img_size,
            'patch_size': args.video_patch_size,
            'encoder_embed_dim': args.video_encoder_embed_dim,
            'encoder_depth': args.video_encoder_depth,
            'encoder_num_heads': args.video_encoder_num_heads,
            'mlp_ratio': args.video_mlp_ratio,
            'qkv_bias': args.video_qkv_bias,
            'encoder_num_classes': args.video_encoder_num_classes,
            'decode_num_classes': args.video_decoder_num_classes,
            'decode_embed_dim': args.video_decoder_embed_dim,
            'decode_num_heads': args.video_decoder_num_heads,
        }
        imu_model_dict = {
            'target_length': args.imu_target_length,
            'plot_type': args.imu_plot_type,
            'plot_height': args.imu_plot_height,
            'patch_size': args.imu_patch_size,
            'channel_num': args.imu_channel_num,
            'encoder_embed_dim': args.imu_encoder_embed_dim,
            'encoder_depth': args.imu_encoder_depth,
            'encoder_num_heads': args.imu_encoder_num_heads,
            'enable_graph': args.imu_enable_graph,
            'imu_graph_net': args.imu_graph_net,
            'imu_two_stream': args.imu_two_stream,
        }
        
        # 모델 생성
        # 모델은 EVIMAE 클래스
        evi_model = models.EVIMAEFT(label_dim=args.n_class, video_model_dict=video_model_dict, imu_model_dict=imu_model_dict)
        evi_model = evi_model.cuda(local_gpu_id)
        evi_model = DDP(module=evi_model,
                        device_ids=[local_gpu_id],
                        find_unused_parameters=True, # 이 부분 검토 필요
                        gradient_as_bucket_view=True)

    else:
        raise ValueError('model not supported')
        
    # initialized with a pretrained checkpoint (adapted from original "video-MAE" checkpoint)
    # 비디오 데이터를 처리하기 위해 video-MAE의 가중치를 활용
    # 사전 학습된 가중치는 모델이 이미 유용한 특징(예: 비디오 데이터의 공간적, 시간적 패턴)을 학습한 상태이므로, 이를 초기화로 사용하면 학습 속도가 빨라지고 성능이 향상될 가능성이 높음
    
    # for the IMU branch, we initialize it with the "ImageMAE" checkpoint, but experiments show that it is not necessary
    # IMU(관성 센서 데이터) 브랜치에 대해서는 ImageMAE(Mask Autoencoder for Images) 모델의 가중치를 사용해 초기화
    # IMU 데이터를 이미지와 유사한 패치 형태로 변환하여 처리하는 경우, ImageMAE의 가중치를 초기화로 사용하는 것이 적절할 수 있음
    # 그런데, 실험 결과 IMU 브랜치에 대해 "사전 학습된 가중치를 사용하지 않고도" 만족스러운 성능을 낼 수 있는듯...?
    if args.pretrain_path != 'None':
        if args.rank == 0:
            if not os.path.isfile(args.pretrain_path):
                raise ValueError('pretrained model not found')
            print(f"Loading pretrained model from {args.pretrain_path}")
            
        # 파일 검사 및 로드 준비 완료 대기
        # 파일 검사 및 로드 준비 이후 모든 Rank가 파일이 유효하다는 것을 확인한 상태에서만 진행하도록 보장
        dist.barrier()  

        # 가중치를 로컬 GPU로 로드
        mdl_weight = torch.load(args.pretrain_path)
        
        model_state_dict = evi_model.state_dict()
        filtered_state_dict = {}
        
        for name, param in mdl_weight.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    filtered_state_dict[name] = param
                else:
                    print(f"Skipped parameter (shape mismatch): {name} - mdl_weight: {param.shape}, evi_model: {model_state_dict[name].shape}")
        
        # 모델 상태 로드
        miss, unexpected = evi_model.load_state_dict(filtered_state_dict, strict=False)
        
        # Rank 0에서만 출력
        if args.rank == 0:
            print('now load mae pretrained weights from ', args.pretrain_path)
            print('miss', miss)
            print('unexpected', unexpected)
            
        # 모델 상태 로드 및 초기화 완료 대기
        # 모델 상태 로드가 모든 Rank에서 완료된 후, 모델 초기 상태가 동기화된 상태에서 다음 작업(예: 학습 시작)으로 진행되도록 보장
        dist.barrier() 

    # Rank 0에서만 디렉토리 생성 및 설정 저장
    if args.rank == 0:
        print("Creating experiment directory: %s" % args.exp_dir)
        try:
            os.makedirs(os.path.join(args.exp_dir, "models"), exist_ok=True)
        except:
            pass
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)
        with open(args.exp_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Rank 0이 작업을 완료한 후 다른 Rank들이 진행할 수 있도록 동기화
    dist.barrier()

    # 모델 학습
    print('Now starting training for {:d} epochs.'.format(args.n_epochs))
    train(evi_model, train_loader, val_loader, train_sampler, args, local_gpu_id)
    
    # Fine-tuning 완료 후, 평균 가중치 모델 처리 부분
    # evaluate with multiple frames

    # 평균 가중치를 사용하는 경우
    # wa_model을 호출하여, 평균화된 모델 가중치를 생성
    if args.rank == 0:
        if args.wa == True:
            sdA = wa_model(args.exp_dir, start_epoch=args.wa_start, end_epoch=args.wa_end, wa_num=args.wa_num)
            torch.save(sdA, args.exp_dir + "/models/evi_model_wa.pth")

        else:
            # if no wa, use the best checkpint
            sdA = torch.load(args.exp_dir + '/models/best_evi_model.pth', map_location='cpu')

    # 모든 Rank가 저장 작업이 완료되기를 기다림
    dist.barrier()

    # rank==0에서만 wandb 종료
    if args.use_wandb and rank == 0:
        wandb.finish()

###########################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser('pretrain wear distributed', parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    
    # set random seed
    seed = args.rseed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    im_res = args.video_img_size
    imu_conf = {'num_mel_bins': 128, 'target_length': args.imu_target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
                'dataset': args.dataset, 'mode':'train', 'mean':args.imu_dataset_mean, 'std':args.imu_dataset_std,
                'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res}
    val_imu_conf = {'num_mel_bins': 128, 'target_length': args.imu_target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                    'mode':'eval', 'mean': args.imu_dataset_mean, 'std': args.imu_dataset_std, 'noise': False, 'im_res': im_res}

    args.world_size = len(args.gpu_ids)
    args.num_workers = len(args.gpu_ids) * 4

    # pretrain evi-mae model
    print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

    # main_worker(args.rank, args)
    mp.spawn(main_worker,
             args=(args, imu_conf, val_imu_conf),
             nprocs=args.world_size,
             join=True)


