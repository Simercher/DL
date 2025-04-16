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
from torch.optim.lr_scheduler import CosineAnnealingLR

patience = 5
save_path = "../saved_models/"
best_val_loss = 999999

def train(args):
    global model_save_dir, save_path, best_val_loss, patience

    # implement the training function here
    train_dataset = load_dataset(args.data_path, "train", args.model_type)  
    valid_dataset = load_dataset(args.data_path, "valid", args.model_type)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader_len = len(train_loader)
    valid_loader_len = len(valid_loader)
    epochs = args.epochs
    dice_scores = []

    # Train the model here
    device = torch.device("mps")
    model = model = Resnet34_Unet() if args.model_type == "resnet34_unet" else Unet()
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Ranger(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice_total = 0.0
        avg_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        for i, sample in enumerate(loop):
            optimizer.zero_grad()
            loss, dice_value = evaluate(model, sample, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 
            train_dice_total += dice_value.item()
            # 更新進度條的 postfix
            loop.set_postfix(loss=loss.item(), dice = dice_value.item())
        scheduler.step()
        avg_train_loss = train_loss / train_loader_len
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Dice: {train_dice_total / train_loader_len:.4f}")
        # Validation
        model.eval()
        valid_loss = 0.0
        val_dice_total = 0.0
        avg_val_loss = 0.0
        loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} - Validating")
        with torch.no_grad():
            for i, sample in enumerate(loop):
                loss, dice_value = evaluate(model, sample, device)
                valid_loss += loss.item()
                val_dice_total += dice_value.item()
                # 更新進度條的 postfix
                loop.set_postfix(loss=loss.item(), dice = dice_value.item())
        avg_val_loss = valid_loss / valid_loader_len
        dice_scores.append(val_dice_total / valid_loader_len)

        with open('Unetlog.txt', 'a') as f:
            f.write(f"Epoch {epoch+1}: Dice = {val_dice_total / valid_loader_len:.4f}\n")
        print(f"Epoch {epoch+1} - Valid Loss: {avg_val_loss:.4f}, Dice: {val_dice_total / valid_loader_len:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            print(f"New best model found. Saving to {save_path}")
            os.makedirs("../saved_models", exist_ok=True)  # 確保資料夾存在
            torch.save(model.state_dict(), save_path+f"/model{epoch+1}.pth")
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    # 畫圖
    import matplotlib.pyplot as plt

    plt.plot(dice_scores)
    plt.title('Dice Coefficient per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.grid(True)
    plt.savefig('dice_curve.png')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--model_type', type=str, choices=['resnet34_unet', 'unet'], default='resnet34_unet', help='Model architecture type')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)