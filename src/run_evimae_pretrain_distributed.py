import argparse
import os
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
from traintest_evimae import train
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)


###########################################################

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    # dataset
    parser.add_argument("--dataset", type=str, default="wear", help="the dataset used", choices=["wear","cmummac", "opp"])
    parser.add_argument("--data-train", type=str, default='', help="training data json")
    parser.add_argument("--data-val", type=str, default='', help="validation data json")
    parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
    parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
    parser.add_argument("--n_class", type=int, default=0, help="number of classes")

    # training
    parser.add_argument("--model", type=str, default='evi-mae', help="the model used")
    parser.add_argument("--noise", help='if use balance sampling', default='True', type=ast.literal_eval)
    parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size') ####?????
    parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
    parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
    parser.add_argument("--lr_adapt", help='if use adaptive learning rate', default='False', type=ast.literal_eval)
    parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
    parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
    parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
    parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
    parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
    parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
    parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
    parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
    parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
    parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
    parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
    parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
    parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default='False')
    parser.add_argument("--load_prepretrain", help='if load pre-pretrain', type=ast.literal_eval, default='True')
    parser.add_argument("--pretrain_modality", type=str, default='imu', help="pretrain modality", choices=['imu', 'video', 'both'])
    parser.add_argument("--image_as_video", help='if use image as video', type=ast.literal_eval, default='False')

    # imu
    parser.add_argument("--imu_target_length", type=int, default=48, help="the target length of imu data")
    parser.add_argument("--imu_masking_ratio", type=float, default=0.75, help="masking ratio")
    parser.add_argument("--imu_mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])
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
    parser.add_argument("--imu_graph_masking_ratio", type=float, default=0.5, help="masking ratio for graph net")

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

    # distributed
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--root', type=str, default='./cifar')

    # wandb
    parser.add_argument("--use_wandb", type=ast.literal_eval, default='True', help="Use wandb logging or not")
    parser.add_argument("--wandb_project", type=str, default='MyProject', help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/team name")

    # seed
    parser.add_argument("--rseed", type=int, default=42, help="random seed")

    return parser


###########################################################


def init_for_distributed(rank, args):

    # 1. setting for distributed training
    # 이 부분에서 rank와 local_gpu_id가 매핑됨
    # rank는 각 프로세스를 의미하고, local_gpu_id는 각 프로세스가 사용할 GPU의 ID 라고 보면 됨
    # 그리고 rank 0은 "마스터 프로세스", rank 1, 2, 3은 "워커 프로세스" 라고 함
    # args.gpu_ids는 [0, 1, 2, 3]
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
    # 즉, print문 앞에 if args.rank == 0: 과 같은 조건을 굳이 붙이지 않아도 된다!
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

    train_set = dataloader.EVIDataset(args.data_train, imu_conf=imu_conf, label_csv=args.label_csv, video_masking_ratio=args.video_masking_ratio, image_as_video=args.image_as_video)
    val_set = dataloader.EVIDataset(args.data_val, imu_conf=val_imu_conf, label_csv=args.label_csv)
    
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

    if args.model == 'evi-mae':
        print('pretrain a evi-mae model with x modality-specific layers and 1 modality-sharing layers')
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
            'masking_ratio': args.video_masking_ratio,
            'pretrain_modality': args.pretrain_modality,
        }

        imu_model_dict = {
            'target_length': args.imu_target_length,
            'masking_ratio': args.imu_masking_ratio,
            'mask_mode': args.imu_mask_mode,
            'plot_type': args.imu_plot_type,
            'plot_height': args.imu_plot_height,
            'patch_size': args.imu_patch_size,
            'channel_num': args.imu_channel_num,
            'encoder_embed_dim': args.imu_encoder_embed_dim,
            'encoder_depth': args.imu_encoder_depth,
            'encoder_num_heads': args.imu_encoder_num_heads,
            'enable_graph': args.imu_enable_graph,
            'imu_graph_net': args.imu_graph_net,
            'imu_graph_masking_ratio': args.imu_graph_masking_ratio,
        }
        
        # 모델 생성
        # 모델은 EVIMAE 클래스
        # find_unused_parameters -> 순전파 중(loss 계산에) 사용되지 않은 파라미터를 감지!!
        # EVIMAE의 forward 함수를 확인하면, 조건문에 의해서 분기가 많이 이루어지고, 이 과정에서 loss 계산에 사용되지 않은 파라미터가 생김... 
        evi_model = models.EVIMAE(norm_pix_loss=args.norm_pix_loss, tr_pos=args.tr_pos, video_model_dict=video_model_dict, imu_model_dict=imu_model_dict)
        evi_model = evi_model.cuda(local_gpu_id)
        evi_model = DDP(module=evi_model,
                        device_ids=[local_gpu_id],
                        find_unused_parameters=True, 
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
    if args.pretrain_path != 'None' and args.load_prepretrain:
        if args.rank == 0:
            if not os.path.isfile(args.pretrain_path):
                raise ValueError('pretrained model not found')
            print(f"Loading pretrained model from {args.pretrain_path}")
            
        # 파일 검사 및 로드 준비 완료 대기
        # 파일 검사 및 로드 준비 이후 모든 Rank가 파일이 유효하다는 것을 확인한 상태에서만 진행하도록 보장
        dist.barrier()  

        # 가중치를 로컬 GPU로 로드
        mdl_weight = torch.load(args.pretrain_path, map_location=torch.device(f'cuda:{local_gpu_id}'))
        
        # 유효한 가중치 필터링
        # 불필요하거나 호환되지 않는 가중치를 제외하고, 새로운 모델에 필요한 가중치만 추출함
        # 키 값에 'graph'가 포함된 가중치를 제외한다!!
        useful_weight = {key: value for key, value in mdl_weight.items() if 'graph' not in key}

        # 모델 상태 로드
        miss, unexpected = evi_model.load_state_dict(useful_weight, strict=False)
        
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
    imu_conf = {'num_mel_bins': 128, 'target_length': args.imu_target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.imu_dataset_mean, 'std':args.imu_dataset_std,
                'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
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

















