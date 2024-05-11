def calculate_metrics(pred, target, threshold=0.5):
    """
    mIoU 및 Accuracy 계산 함수.
    
    Args:
        pred (torch.Tensor): 모델의 예측값.
        target (torch.Tensor): 실제 레이블.
        threshold (float): 이진화 임계값.
        
    Returns:
        (tuple): mIoU와 Accuracy의 튜플.
    """
    pred_binary = (pred > threshold).int()
    target_binary = (target > 0.5).int()
    intersection = (pred_binary & target_binary).float().sum((1, 2, 3))
    union = (pred_binary | target_binary).float().sum((1, 2, 3))

    iou = (intersection / (union + 1e-6)).mean().item()
    accuracy = (pred_binary == target_binary).float().mean().item()

    return iou, accuracy
