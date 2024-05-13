from scipy.spatial.distance import directed_hausdorff
from monai.metrics import DiceMetric


def train_metrics(preds, targets, threshold=0.5):
    """
    mIoU 및 Accuracy 계산 함수.
    
    Args:
        preds (torch.Tensor): 모델의 예측값.
        targets (torch.Tensor): 실제 레이블.
        threshold (float): 이진화 임계값.
        
    Returns:
        (tuple): mIoU와 Accuracy, Dice Score, Hausdorff Distance Metric의 튜플.
    """
    preds_binary = (preds > threshold).int()
    targets_binary = targets.clone().int()
    intersection = (preds_binary & targets_binary).sum((1, 2, 3))
    union = (preds_binary | targets_binary).sum((1, 2, 3))

    # Calculate metrics
    iou = (intersection / (union + 1e-6)).mean().item()
    accuracy = (preds_binary == targets_binary).float().mean().item()

    return iou, accuracy


def calculate_metrics(preds, targets, threshold=0.5):
    """
    mIoU 및 Accuracy 계산 함수.
    
    Args:
        preds (torch.Tensor): 모델의 예측값.
        targets (torch.Tensor): 실제 레이블.
        threshold (float): 이진화 임계값.
        
    Returns:
        (tuple): mIoU와 Accuracy, Dice Score, Hausdorff Distance Metric의 튜플.
    """
    preds_binary = (preds > threshold).int()
    targets_binary = targets.clone().int()
    intersection = (preds_binary & targets_binary).sum((1, 2, 3))
    union = (preds_binary | targets_binary).sum((1, 2, 3))

    # Calculate metrics
    iou = (intersection / (union + 1e-6)).mean().item()
    accuracy = (preds_binary == targets_binary).float().mean().item()

    # Dice Score
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    dice_score = dice_metric(preds_binary, targets_binary).mean()

    hausdorff_distance = calculate_symmetric_hausdorff(preds_binary.cpu().numpy(), targets_binary.cpu().numpy())
    return iou, accuracy, dice_score, hausdorff_distance

def calculate_symmetric_hausdorff(preds, targets):
    """ 대칭적인 Hausdorff 거리를 얻기 위해 두 방향 모두를 계산하고 최대값을 사용 """
    hausdorff_distances = []
    for i in range(preds.shape[0]):  # Loop over the batch
        u = preds[i].reshape(-1, preds.shape[2] * preds.shape[3])  # Reshape from C, H, W to C, H*W
        v = targets[i].reshape(-1, targets.shape[2] * targets.shape[3])
        forward = directed_hausdorff(u, v)[0]
        backward = directed_hausdorff(v, u)[0]
        hausdorff_distances.append(max(forward, backward))
    return max(hausdorff_distances)  # Return the maximum across the batch

