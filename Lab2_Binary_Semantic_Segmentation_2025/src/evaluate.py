import torch
import utils


def tverskyLoss(y_pred, y_true):
    """
    y_pred: (batch, C, H, W)  預測輸出 (sigmoid or softmax)
    y_true: (batch, C, H, W)  標籤 (0/1)
    """

    smooth = 1e-6
    alpha = 0.7
    beta = 0.3

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    TP = (y_true * y_pred).sum()
    FP = ((1 - y_true) * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()

    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    return 1 - tversky_index
def MCCLoss(probs, targets):
    # 計算 TP、TN、FP、FN
    eps=1e-6
    TP = (probs * targets).sum()
    TN = ((1 - probs) * (1 - targets)).sum()
    FP = (probs * (1 - targets)).sum()
    FN = ((1 - probs) * targets).sum()

    # 計算 MCC
    numerator = TP * TN - FP * FN
    denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps)
    
    mcc = numerator / (denominator + eps)

    # MCC Loss: 1 - MCC (MCC 越高 Loss 越低)
    return 1 - mcc

def evaluate(model, data, device):
    images, masks = data["image"].to(device), data["mask"].to(device)
    outputs = model(images)
    sigmoid_outputs = torch.sigmoid(outputs) # sigmoid activation function for binary output
    dice_value = utils.dice_score(sigmoid_outputs.detach(), masks)
    tversky_loss = tverskyLoss(sigmoid_outputs, masks)
    mccloss = MCCLoss(sigmoid_outputs, masks)
    loss = tversky_loss + (1 - dice_value) + mccloss
    return loss, dice_value