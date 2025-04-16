import torch

def dice_score(preds, targets, threshold=0.5, epsilon=1e-6):

    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2. * intersection + epsilon) / (union)
    return dice.mean()

def tverskyLoss(y_pred, y_true):
    """
    y_pred: (batch, C, H, W)  預測輸出 (sigmoid or softmax)
    y_true: (batch, C, H, W)  標籤 (0/1)
    """

    smooth = 1e-6
    alpha = 0.5
    beta = 0.5

    y_pred = y_pred.view(-1)  # 展平
    y_true = y_true.view(-1)

    TP = (y_true * y_pred).sum()
    FP = ((1 - y_true) * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()

    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    return 1 - tversky_index  # 損失值
