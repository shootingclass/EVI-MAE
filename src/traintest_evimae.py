# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import torch.distributed as dist

import wandb

def train(evi_model, train_loader, test_loader, train_sampler, args, local_gpu_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_g_meter = AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir


    # _로 내부 함수임을 표시
    # rank 0만 저장 -> rank 0 프로세스에서만 실행되는 동작에 대해서는 dist.barrier를 호출할 필요가 없음!!
    # rank 0에서만 해당 동작을 수행하므로, 다른 프로세스와의 동기화가 필요하지 않기 때문...
    def _save_progress():        
        if args.rank == 0:        
            progress.append([epoch, global_step, best_epoch, best_loss, time.time() - start_time])
            with open("%s/progress.pkl" % exp_dir, "wb") as f:
                pickle.dump(progress, f)


    trainables = [p for p in evi_model.parameters() if p.requires_grad]
    
    print('Total parameter number is : {:.3f}'.format(sum(p.numel() for p in evi_model.parameters())))
    print('Total trainable parameter number is : {:.3f}'.format(sum(p.numel() for p in trainables)))
    
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else: # pretrain here
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(evi_model.state_dict(), "%s/models/evi_model.%d.pth" % (exp_dir, epoch))

    epoch += 1
    scaler = GradScaler()

    print("start training...")
    
    result = np.zeros([args.n_epochs, 12])  # for each epoch, 10 metrics to record
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        evi_model.train()
        
        # 매 epoch 시작 전 set_epoch을 호출해야 shuffle이 됨!!
        train_sampler.set_epoch(epoch)
        
        # 모든 프로세스가 동일한 epoch을 시작하도록 동기화
        dist.barrier()  
        
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        # Training
        for i, (a_input, v_input, v_mask, _) in enumerate(train_loader): 
            
            # a_input -> fbank, 전처리된 spectrogram
            # v_input -> process_data, 비디오 데이터를 모델 학습에 사용 가능한 텐서(C, T, H, W)로 정제한 결과
            # v_mask -> datum['video_path'], 'trim_5s_400x300_correct60/sbj_17_1470_1475.mp4' 과 같이 그냥 경로임
            # label_indices -> _, 레이블
            
            # print(a_input.shape, v_input.shape, v_mask.shape)
            # [b, 12, 48, 64], [b, 3, 16, 224, 224], [b, 1568]
            
            B = a_input.size(0)
            a_input = a_input.to(local_gpu_id, non_blocking=True)
            v_input = v_input.to(local_gpu_id, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()


            # # 원본 코드
            # with autocast():
            #     loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g = evi_model(a_input, v_input, v_mask, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.imu_mask_mode)
            
            #     # 각 GPU에서 계산된 loss를 합산
            #     # 각 GPU에서 계산된 정확도는 평균을 냄
            
            #     # torch.nn.DataParallel을 사용할 때, 입력 데이터는 자동으로 여러 GPU에 분배되고 각 GPU에서 독립적으로 연산이 수행됨
            #     # 4개의 GPU를 사용하고, 배치 크기가 32라면, 각 GPU는 8개의 샘플을 처리하게 됨
            #     # 이때, 각 GPU에서 계산된 loss를 sum()을 이용하여 합산하는거임
            #     loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_g = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean(), loss_g.sum()


            with autocast():
                
                # 여기서 EVIMAE 클래스의 forward 함수 호출
                # 각 GPU에서 loss, loss_mae 등 개별적으로 계산
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g = evi_model(a_input, v_input, v_mask, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.imu_mask_mode)
                
                # 모든 프로세스 동기화 (진입 확인)
                dist.barrier()  
                
                # 손실 값들 합산
                # loss.backward() 이전에, GPU별로 계산된 손실 값을 동기화 하고자 할 때 dist.all_reduce 이용
                # 그래디언트 동기화와는 별개로, 손실 값의 평균이나 다른 통계를 모든 GPU에서 동일하게 기록하고자 할 때 필요
                # 결국 dist.all_reduce는 각 GPU에서 계산된 값을 합산하거나 특정 연산(SUM, MEAN 등)을 수행하여 모든 GPU가 동일한 결과를 가지도록 함
                # 그래서 왜 사용하는가? -> 분산 학습에서는 각 GPU가 서로 다른 데이터 샘플을 처리함. 따라서, 손실 값을 동기화하지 않으면 GPU마다 다른 손실 값을 가지게 되어 전체 학습이 일관되지 않게 됨...
                
                # 여러 스칼라 값을 하나의 텐서로 결합
                losses = torch.stack([loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, loss_g])
                
                # 모든 GPU의 losses 텐서 값을 합산
                # 각 GPU에서 계산된 손실 값이 다를 수 있으므로, all_reduce를 통해 각 GPU의 손실 값을 모두 합산한 결과를 "모든 GPU가 동일하게" 가지게 함
                # GPU 0: [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
                # GPU 1: [0.8, 0.4, 0.2, 0.1, 0.05, 0.02]
                # all_reduce 결과 (SUM): [1.8, 0.9, 0.5, 0.3, 0.15, 0.07]
                dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                
                # 합산된 결과를 다시 각 손실 값에 할당
                # 모든 GPU는 동일한 losses 텐서를 가지며, 각각의 값을 원래의 손실 변수(loss, loss_mae, ...)에 저장함
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, loss_g = losses 
                
                # c_acc는 정확도에 해당하므로, 평균을 구함
                dist.all_reduce(c_acc, op=dist.ReduceOp.SUM)
                c_acc = c_acc / args.world_size 

            optimizer.zero_grad()                    
            scaler.scale(loss).backward()
            
            # 모든 랭크가 backward 완료를 대기            
            dist.barrier()  
            
            scaler.step(optimizer)            
            scaler.update()

            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            loss_g_meter.update(loss_g.item(), B)            
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if args.rank == 0:
                if print_step and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                    'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                    'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                    'Train Total Loss {loss_av_meter.val:.4f}\t'
                    'Train MAE Loss imu {loss_a_meter.val:.4f}\t'
                    'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                    'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                    'Train Contrastive Acc {c_acc:.3f}\t'
                    'Train Graph Loss {loss_g_meter.val:.4f}\t'
                    .format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                        per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc, loss_g_meter=loss_g_meter), flush=True)

            end_time = time.time()
            global_step += 1

            # (2) 일정 간격마다 Train Loss wandb 로깅
            #     rank=0(마스터 프로세스)일 때만
            if (global_step % args.n_print_steps == 0) and args.rank == 0 and args.use_wandb:
                wandb.log({"train/total_loss": loss_av_meter.val,
                           "train/mae_loss_imu": loss_a_meter.val,
                           "train/mae_loss_visual": loss_v_meter.val,
                           "train/contrastive_loss": loss_c_meter.val,
                           "train/contrastive_acc": c_acc,
                           "train_graph_loss": loss_g_meter.val,
                            "train/epoch": epoch, 
                            "train/step": global_step})                


        # 모든 프로세스가 훈련을 완료했음을 보장
        # Validation 전에 동기화 추가
        dist.barrier()
        
        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc, eval_loss_g = validate(evi_model, test_loader, args, local_gpu_id)

        # 수정 필요
        if args.rank == 0 and args.use_wandb:
            wandb.log({
                "val/total_loss": eval_loss_av,
                "val/mae_loss": eval_loss_mae, 
                "val/mae_loss_imu": eval_loss_mae_a,
                "val/mae_loss_visual": eval_loss_mae_v,
                "val/contrastive_loss": eval_loss_c,
                "val/contrastive_acc": eval_c_acc,
                "val/graph_loss": eval_loss_g,
                "epoch": epoch
            }, step=global_step)

        # train imu mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval imu mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr'], loss_g_meter.avg, eval_loss_g]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if args.rank == 0:
            
            # Best 모델 저장
            if eval_loss_av < best_loss:
                best_loss = eval_loss_av
                best_epoch = epoch

            if best_epoch == epoch:
                
                # 모델의 가중치 저장
                # 학습 가능한 모든 파라미터 저장
                # 모델의 가중치를 불러와서 검증, 테스트, 또는 Fine-Tuning에 사용
                # 즉, Fine Tuning을 하려면 모델 가중치만 로드하면 된다
                torch.save(evi_model.state_dict(), "%s/models/best_evi_model.pth" % (exp_dir))
                
                # 최적화기 상태 저장
                # 각 파라미터에 대한 learning rate, 모멘텀, weight_decay 등 저장
                # 학습이 중단된 시점에서, 동일한 상태로 이어서 학습을 수행할 때 필요
                # 즉, 중단된 학습을 이어서 진행하려는 경우, 모델 가중치와 최적화기 상태를 모두 로드해야 함
                torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

            # 특정 epoch에 도달했을 때 모델 저장
            if args.save_model == True:
                if epoch in [50, 100, 150, 200, 250, 299]:
                    torch.save(evi_model.state_dict(), "%s/models/evi_model.%d.pth" % (exp_dir, epoch))

            # scheduler step
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(-eval_loss_av)
            else:
                scheduler.step()

            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

            _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))
        
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()
        loss_g_meter.reset()
        
        epoch += 1
        
        # 모델 저장 후의 barrier
        # 마스터 프로세스가 모델 저장을 완전히 완료할 때까지 다른 프로세스들이 기다리게 함
        # 모든 프로세스가 리셋 작업을 동시에 시작하도록 보장
        # 한 epoch의 완전한 종료와 다음 epoch의 시작을 명확히 구분
        dist.barrier()


def validate(evi_model, val_loader, args, local_gpu_id):
    batch_time = AverageMeter()
    evi_model.eval()
    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc, A_loss_g = [], [], [], [], [], [], []
   
    with torch.no_grad():
        for i, (a_input, v_input, v_mask, _) in enumerate(val_loader):
            a_input = a_input.to(local_gpu_id, non_blocking=True)
            v_input = v_input.to(local_gpu_id, non_blocking=True)
   
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g = evi_model(a_input, v_input, v_mask, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.imu_mask_mode)
                    
                dist.barrier()    
                
                # 손실 값들 합산
                losses = torch.stack([loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, loss_g])
                dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, loss_g = losses       
                
                # c_acc는 정확도에 해당하므로, 평균을 구함
                dist.all_reduce(c_acc, op=dist.ReduceOp.SUM)
                c_acc = c_acc / args.world_size                  
   
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            A_loss_g.append(loss_g.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)
        loss_g = np.mean(A_loss_g)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_g