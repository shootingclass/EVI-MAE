import numpy as np
from scipy import stats
from sklearn import metrics
import torch
import torch.distributed as dist


def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


# dist.all_gather
# 각 Rank(프로세스)에서 가지고 있는 텐서를 모든 Rank에 모으는 함수
# Rank 0이 x0, Rank 1이 x1, …, Rank N이 xN을 가지고 있다고 하면, all_gather 후에는 모든 Rank가 [x0, x1, …, xN] 전체를 얻게 됨

# dist.all_reduce는 모든 Rank에서 가지고 있는 텐서 값을 특정 연산(합, 평균, 최댓값 등)을 통해 집계한 뒤, 그 결과를 모든 Rank에 동일하게 다시 전달하는 함수
# 결국 분산 학습 시, 데이터 그 자체를 모아야 하는 상황(예: 최종 평가에 필요한 예측 결과)은 all_gather를 사용
# 집계 연산 결과만 필요(예: 전체 손실 합, 정확도 합)하면 all_reduce를 사용
def gather_all_data_gpu(tensor, local_gpu_id):
    """
    1) 텐서를 local_gpu_id 장치로 이동.
    2) dist.all_gather로 모든 Rank의 텐서를 모은 뒤 하나로 concat.
    3) 최종적으로 모든 Rank가 동일한 전체 텐서를 반환받는다.
    """
    # 텐서를 GPU로 이동
    device = torch.device(f"cuda:{local_gpu_id}")
    tensor = tensor.to(device)

    # gather_list에 각 Rank에서 온 텐서를 저장할 공간을 마련
    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

    # 모든 Rank의 텐서를 gather_list에 모은다
    dist.all_gather(gather_list, tensor)

    # 모든 Rank가 동일한 데이터를 얻게 되므로, cat으로 하나의 텐서로 합침
    gathered_tensor = torch.cat(gather_list, dim=0)
    return gathered_tensor


def calculate_stats(output, target, vpaths, local_gpu_id):
    """
    Args:
        output: (batch_size, classes_num) 형태의 텐서 (CPU/GPU 어느 쪽이든 가능)
        target: (batch_size, classes_num) 형태의 텐서 (CPU/GPU)
        vpaths: (optional) 파일 경로 등 문자열 목록
        local_gpu_id: 현재 Rank의 GPU ID (int)
    """

    # 1) 모든 Rank에서 output, target 텐서를 GPU로 모아서 합친다.
    output = gather_all_data_gpu(output, local_gpu_id)  # GPU 텐서 (N, classes_num)
    target = gather_all_data_gpu(target, local_gpu_id)  # GPU 텐서 (N, classes_num)

    # 2) 지금부터는 output, target은 하나로 합쳐진 "전체 데이터"가 GPU에 있음.
    #    클래스가 여러 개인 단일 라벨 분류를 가정.

    # (예) 다중 클래스 단일 라벨 분류에서 argmax를 이용한 정답 추론
    pred_indices = output.argmax(dim=1)   # PyTorch 텐서 (N,)
    true_indices = target.argmax(dim=1)   # PyTorch 텐서 (N,)

    # 3) 클래스 존재 여부 판단 등은 PyTorch 연산 사용
    #    예: "특정 클래스에 양성 레이블이 하나도 없으면 건너뛴다"
    #    -> 나중에 "각 클래스별"로 torch.sum(target[:, k]) == 0 체크

    # 4) 이제 Accuracy 등 Scikit-Learn 지표 계산 직전에만 NumPy 변환
    pred_indices_np = pred_indices.cpu().numpy()
    true_indices_np = true_indices.cpu().numpy()

    acc = metrics.accuracy_score(true_indices_np, pred_indices_np)

    classes_num = target.shape[1]
    stats = []

    # 5) 각 클래스를 반복하면서 AP, AUC 등을 계산
    for k in range(classes_num):
        # (A) PyTorch 텐서로 해당 클래스의 레이블 존재 여부 확인
        #     -> 레이블이 전혀 없으면 평균 정밀도(AP)나 AUC 계산 불가
        if torch.sum(target[:, k]) == 0:
            # target[:, k].sum() == 0이라면 해당 클래스 샘플이 없으므로 건너뜀
            continue

        # (B) 지표 계산 전 NumPy로 변환
        target_k_np = target[:, k].cpu().numpy()
        output_k_np = output[:, k].cpu().numpy()

        avg_precision = metrics.average_precision_score(target_k_np, output_k_np, average=None)

        try:
            auc = metrics.roc_auc_score(target_k_np, output_k_np, average=None)
            precisions, recalls, _ = metrics.precision_recall_curve(target_k_np, output_k_np)
            fpr, tpr, _ = metrics.roc_curve(target_k_np, output_k_np)

            stats_dict = {
                'precisions': precisions[0::1000],
                'recalls':    recalls[0::1000],
                'AP':         avg_precision,
                'fpr':        fpr[0::1000],
                'fnr':        1. - tpr[0::1000],
                'auc':        auc,
                'acc':        acc  # 전체 Accuracy(클래스별 x)
            }
        except:
            stats_dict = {
                'precisions': -1,
                'recalls':    -1,
                'AP':         avg_precision,
                'fpr':        -1,
                'fnr':        -1,
                'auc':        -1,
                'acc':        acc
            }
            print(f"Class {k}: no true sample or error in metrics.")

        stats.append(stats_dict)

    return stats