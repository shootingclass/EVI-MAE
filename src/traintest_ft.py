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

    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir


    # _로 내부 함수임을 표시
    def _save_progress():        
        # rank 0만 저장
        if args.rank == 0:        
            progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
            with open("%s/progress.pkl" % exp_dir, "wb") as f:
                pickle.dump(progress, f)
        
        # 모든 Rank 동기화
        dist.barrier()


    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'another_mlp_head.0.weight', 'another_mlp_head.0.bias', 'another_mlp_head.1.weight', 'another_mlp_head.1.bias',
                'mlp_head2.0.weight', 'mlp_head2.0.bias', 'mlp_head2.1.weight', 'mlp_head2.1.bias',
                'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
                'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
                'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, evi_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, evi_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if args.freeze_base == True: # False
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False


    trainables = [p for p in evi_model.parameters() if p.requires_grad]

    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in evi_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
    print('The other base layers use {:.3f} x larger lr'.format(args.base_lr))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr * args.base_lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # only for preliminary test, formal exps should use fixed learning rate scheduler
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    
    main_metrics = args.metrics
    
    # args.loss == BCE 이므로, loss_fn은 nn.BCEWithLogitsLoss()
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss().to(local_gpu_id)
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(local_gpu_id)
    
    args.loss_fn = loss_fn
    loss_two_stream = nn.MSELoss()

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    
    result = np.zeros([args.n_epochs, 4])

    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()

        if args.only_val == False:
            evi_model.train()
            train_sampler.set_epoch(epoch)
            
            print('---------------')
            print(datetime.datetime.now())
            print("current #epochs=%s, #steps=%s" % (epoch, global_step))

            # Training
            for i, (a_input, v_input, _, labels) in enumerate(train_loader):

                B = a_input.size(0)
                a_input, v_input = a_input.to(local_gpu_id, non_blocking=True), v_input.to(local_gpu_id, non_blocking=True)
                labels = labels.to(local_gpu_id, non_blocking=True)

                data_time.update(time.time() - end_time)
                per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
                dnn_start_time = time.time()

                with autocast():
                    
                    # EVIMAEFT 모델의 forward 함수 호출
                    imu_output = evi_model(a_input, v_input, args.ftmode)

                    # imu_enable_graph는 True, imu_two_stream은 False
                    # if args.imu_enable_graph and args.imu_two_stream:
                    #     output1, output2 = imu_output
                    #     imu_output = output1
                    #     another_loss1 = loss_fn(output2, labels)
                    #     # detach the output1 to avoid backpropagation
                    #     output1_clone_detach = output1.clone().detach()
                    #     another_loss2 = loss_two_stream(output1_clone_detach, output2)
                        
                    loss = loss_fn(imu_output, labels)
                    
                    # if args.imu_enable_graph and args.imu_two_stream:
                    #     print(loss, another_loss1, another_loss2)
                    #     loss += another_loss1
                    #     loss += another_loss2 * 0.1
                        

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), B)
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
                        'Train Loss {loss_meter.val:.4f}\t'.format(
                        epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                            per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)

                end_time = time.time()
                global_step += 1

                # (2) 일정 간격마다 Train Loss wandb 로깅
                #     rank=0(마스터 프로세스)일 때만
                if (global_step % args.n_print_steps == 0) and args.rank == 0 and args.use_wandb:
                    wandb.log({"train/loss": loss_meter.val, 
                            "train/epoch": epoch, 
                            "train/step": global_step})

        print('start validation')
        stats, valid_loss = validate(evi_model, test_loader, args, local_gpu_id, output_pred=False)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        if args.rank == 0 and args.use_wandb:
            wandb.log({
                "train/loss": loss_meter.avg,
                "val/loss": valid_loss,
                "val/mAP": mAP,
                "val/mAUC": mAUC,
                "val/acc": acc,
                "val/d_prime": d_prime(mAUC),
                "epoch": epoch,
            }, step=global_step)

        # print("train_loss: {:.6f}".format(loss_meter.avg))
        # print("valid_loss: {:.6f}".format(valid_loss))
        # print("mAP: {:.6f}".format(mAP))
        # print("AUC: {:.6f}".format(mAUC))
        # print("acc: {:.6f}".format(acc))
        # print("d_prime: {:.6f}".format(d_prime(mAUC)))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if args.rank == 0:
            
            # Best 모델 저장
            if mAP > best_mAP:
                best_mAP = mAP
                if main_metrics == 'mAP':
                    best_epoch = epoch

            if acc > best_acc:
                best_acc = acc
                if main_metrics == 'acc':
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
                if epoch in [50,100,150,199]:
                    torch.save(evi_model.state_dict(), "%s/models/evi_model.%d.pth" % (exp_dir, epoch))


        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()



def validate(evi_model, val_loader, args, local_gpu_id, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    evi_model.eval()
    end = time.time()
    A_predictions, A_targets, A_loss, A_path = [], [], [], []
    
    with torch.no_grad():
        for i, (a_input, v_input, vpath, labels) in enumerate(val_loader):
            a_input = a_input.to(local_gpu_id, non_blocking=True)
            v_input = v_input.to(local_gpu_id, non_blocking=True)

            with autocast():
                imu_output = evi_model(a_input, v_input, args.ftmode)
                
                # if args.imu_enable_graph and args.imu_two_stream:
                #     output1, output2 = imu_output
                #     imu_output = output1

            predictions = imu_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)
            A_path.append(vpath)

            labels = labels.to(local_gpu_id, non_blocking=True)
            
            loss = args.loss_fn(imu_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        imu_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        vpaths = [item for sublist in A_path for item in sublist]
        loss = np.mean(A_loss)

        stats = calculate_stats(imu_output, target, vpaths, local_gpu_id)

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, imu_output, target