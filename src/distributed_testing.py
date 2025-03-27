import os
import time
import torch
import visdom
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

# for dataset
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# for model
from torchvision.models import vgg11
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    # parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--root', type=str, default='./cifar')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./save')
    parser.add_argument('--save_file_name', type=str, default='vgg_cifar')
    # usage : --gpu_ids 0, 1, 2, 3
    return parser


def main_worker(rank, opts):
	# 1. argparse (main)
    # 2. init dist
    local_gpu_id = init_for_distributed(rank, opts)    
    
    # 4. data set
    transform_train = tfs.Compose([
        tfs.Resize(256),
        tfs.RandomCrop(224),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tfs.Compose([
                                  tfs.Resize(256),
                                  tfs.CenterCrop(224),
                                  tfs.ToTensor(),
                                  tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                std=(0.2023, 0.1994, 0.2010)),
                                        ])

    train_set = CIFAR10(root=opts.root,
                        train=True,
                        transform=transform_train,
                        download=True)

    test_set = CIFAR10(root=opts.root,
                       train=False,
                       transform=transform_test,
                       download=True)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=int(opts.batch_size / opts.world_size),
                              shuffle=False,
                              num_workers=int(opts.num_workers / opts.world_size),
                              sampler=train_sampler,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=int(opts.batch_size / opts.world_size),
                             shuffle=False,
                             num_workers=int(opts.num_workers / opts.world_size),
                             sampler=test_sampler,
                             pin_memory=True)

    # 5. model
    model = vgg11(weights=None)
    model = model.cuda(local_gpu_id)
    model = DDP(module=model,
                device_ids=[local_gpu_id])

    # 6. criterion
    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.01,
                                weight_decay=0.0005,
                                momentum=0.9)

    # 8. scheduler
    scheduler = StepLR(optimizer=optimizer,
                       step_size=30,
                       gamma=0.1)


    # 중단된 학습 복원
    # if opts.start_epoch != 0:

    #     checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
    #                             .format(opts.start_epoch - 1),
    #                             map_location=torch.device('cuda:{}'.format(local_gpu_id)))
    #     model.load_state_dict(checkpoint['model_state_dict'])  # load model state dict
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # load optim state dict
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # load sched state dict
        
    #     if opts.rank == 0:
    #         print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))


    for epoch in range(opts.start_epoch, opts.epoch):

        # 9. train
        tic = time.time()
        
        # 학습 모드로 설정
        model.train()
        
        # DistributedSampler는 분산 학습 시 데이터셋을 프로세스별로 나누어 샘플링하도록 설계된 샘플러
        # 각 프로세스가 서로 다른 부분의 데이터를 처리하여 데이터 중복을 방지하고 학습 효율성을 높임
        # DistributedSampler는 데이터셋의 순서를 **난수(random seed)**로 섞어 데이터 다양성을 제공
        # set_epoch(epoch)는 에포크별로 난수를 고정하여 모든 프로세스가 동일한 순서로 데이터 샘플링을 진행하도록 보장
        # 이를 통해 데이터 중복 방지와 동기화를 실현
        # set_epoch(epoch)를 호출하면, 주어진 에포크 번호(epoch)를 기반으로 새로운 난수를 생성하여 데이터 순서를 재설정
        # 난수의 고정은 에포크마다 동일한 데이터 순서로 동작하게 하고, 분산 학습 환경에서도 데이터가 잘 섞이도록 보장
        train_sampler.set_epoch(epoch) 

        # 학습 단계에서, local_gpu_id는 각 프로세스에 할당된 GPU의 ID
        # train 함수에서 DistributedDataParallel (DDP)는 각 프로세스가 자신에게 할당된 GPU에서만 작업을 수행하도록 설정되므로, 데이터를 해당 GPU로 명시적으로 옮김
        for i, (images, labels) in enumerate(train_loader):            
            
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images)

            # ----------- update -----------
            optimizer.zero_grad()
            
            # loss 계산 시에 dist.all_reduce가 사용되지 않았음
            # PyTorch의 DistributedDataParallel은 모델의 모든 파라미터에 대해 backward pass 동안 dist.all_reduce를 호출하여 그래디언트를 자동으로 동기화하기 때문
            # 따라서, 사용자가 loss.backward()를 호출하면 DDP가 각 GPU에서 계산된 그래디언트를 자동으로 집계(sum)한 뒤, 이를 각 GPU에 다시 나누어줍
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # get lr
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # time
            toc = time.time()

        # save pth file
        # Master 프로세스(rank == 0)만 체크포인트 저장
        # 분산 학습에서 모든 프로세스가 동일한 작업을 수행하므로, 체크포인트 저장은 중복을 방지하기 위해 Master 프로세스에서만 수행행
        if opts.rank == 0:
            if not os.path.exists(opts.save_path):
                os.mkdir(opts.save_path)

            checkpoint = {'epoch': epoch, # 현재 epoch
                          'model_state_dict': model.state_dict(), # 모델 파라미터
                          'optimizer_state_dict': optimizer.state_dict(), # 옵티마이저 상태
                          'scheduler_state_dict': scheduler.state_dict()} # lr scheduler 상태

            torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))
            print("save pth.tar {} epoch!".format(epoch))

        # 10. test
        
        # 분산 학습 환경에서 검증은 보통 Master 프로세스(rank == 0)에서만 수행함!!
        if opts.rank == 0:
            
            # 평가 모드로 전환
            model.eval()

            val_avg_loss = 0
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    images = images.to(opts.rank)  # [100, 3, 224, 224]
                    labels = labels.to(opts.rank)  # [100]
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_avg_loss += loss.item()
                    # ------------------------------------------------------------------------------
                    # rank 1
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct_top1 += (pred == labels).sum().item()

                    # ------------------------------------------------------------------------------
                    # rank 5
                    _, rank5 = outputs.topk(5, 1, True, True)
                    rank5 = rank5.t()
                    correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

                    # ------------------------------------------------------------------------------
                    for k in range(5):  # 0, 1, 2, 3, 4, 5
                        correct_k = correct5[:k+1].reshape(-1).float().sum(0, keepdim=True)
                    correct_top5 += correct_k.item()

            accuracy_top1 = correct_top1 / total
            accuracy_top5 = correct_top5 / total

            val_avg_loss = val_avg_loss / len(test_loader)  # make mean loss

            print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
            print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))
            scheduler.step()

    return 0


def init_for_distributed(rank, opts):
    
    # 1. setting for distributed training
    # 이 부분에서 rank와 local_gpu_id가 매핑됨
    # rank는 각 프로세스를 의미하고, local_gpu_id는 각 프로세스가 사용할 GPU의 ID 라고 보면 됨
    # 그리고 rank 0은 "마스터 프로세스", rank 1, 2, 3은 "워커 프로세스" 라고 함
    # opts.gpu_ids는 [0, 1, 2, 3]
    # 결국 rank 0 -> local_gpu_id 0, .... rank 3 -> local_gpu_id 3 까지 1:1로 매핑됨
    # 그 후, 해당 GPU가 현재 프로세스의 기본 디바이스로 설정됨
    opts.rank = rank    
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    # 모든 프로세스가 이 지점에 도달할 때까지 대기, 프로세스들의 동기화를 보장
    torch.distributed.barrier()
    
    # convert print fn if rank is zero
    # rank가 0일 때에만 print가 이루어짐!! 여러 프로세스가 동시에 print문을 실행하면 콘솔이 복잡해지니까...
    # 즉, print문 앞에 if opts.rank == 0: 과 같은 조건을 굳이 붙이지 않아도 된다!
    setup_for_distributed(opts.rank == 0)
    print(opts)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser('vgg11 cifar training', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    # mp.spawn은 병렬로 여러 프로세스를 생성, 각 프로세스마다 main_worker 함수를 실행하면서 다른 rank 값을 전달
    # main_worker(rank=0, opts)  # 첫 번째 프로세스
    # main_worker(rank=1, opts)  # 두 번째 프로세스
    # main_worker(rank=2, opts)  # 세 번째 프로세스
    # main_worker(rank=3, opts)  # 네 번째 프로세스
    # main_worker(opts.rank, opts)
    # 여기서 rank 값이 init_for_distributed 함수를 통해 각 프로세스의 GPU 할당에 사용됨
    mp.spawn(main_worker,
             args=(opts, ),
             nprocs=opts.world_size,
             join=True)
    
    
    
# python distributed_testing.py 