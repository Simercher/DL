import argparse
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import Unet
from models.resnet34_unet import Resnet34_Unet
import torch
from tqdm import tqdm
import utils
import os
import numpy as np
from evaluate import evaluate
import torch_optimizer as optim

def test_model(args):
    test_dataset = load_dataset(args.data_path, "test", args.model_type)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 
    test_loader_len = len(test_loader)
    dice_scores = []

    # Train the model here
    device = torch.device("mps")
    model = Resnet34_Unet() if args.model_type == "resnet34_unet" else Unet()
    model = model.to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded model weights from {args.model}")
    else:
        raise FileNotFoundError(f"Model weight file {args.model} not found!")
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.eval()
    test_loss = 0.0
    test_dice_total = 0.0
    avg_test_loss = 0.0
    last_image, last_output, last_mask = None, None, None
    loop = tqdm(test_loader, desc=f" - Testing")
    with torch.no_grad():
        for i, sample in enumerate(loop):
            loss, dice_value = evaluate(model, sample, device)
            test_loss += loss.item()
            test_dice_total += dice_value.item()
            # 更新進度條的 postfix
            loop.set_postfix(loss=loss.item(), dice = dice_value.item())
    avg_test_loss = test_loss / test_loader_len
    dice_scores.append(test_dice_total / test_loader_len)
    print(f"Test Loss: {avg_test_loss:.4f}, Dice: {test_dice_total / test_loader_len:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--model_type', type=str, choices=['resnet34_unet', 'unet'], default='resnet34_unet', help='Model architecture type')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    test_model(args)

    # assert False, "Not implemented yet!"