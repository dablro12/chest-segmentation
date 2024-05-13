#!/usr/bin/env python
import sys 
sys.path.append('../')
import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms 
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import Segmentation_CustomDataset as CustomDataset
from utils.metrics import train_metrics
from utils.__init__ import *
from utils.arg import save_args
from utils import * 

import wandb
# Set environment variable for compatibility issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Disable unnecessary precision to speed up computations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(False) 
#################
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.dataset import Segmentation_CustomDataset as CustomDataset
import torch.optim as optim
from utils.__init__ import *
from utils import * 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from model import load_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Train(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.setup_device(args)
        self.setup_datasets(args)
        self.setup_wandb(args)
        self.initialize_models(args)
        self.setup_train(args)
        self.setup_paths(args)
        

        self.best_valid_loss = np.inf
        self.epochs_no_improve = 0
        self.n_epochs_stop = 50  # Number of epochs to wait before stopping

    def setup_device(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CUDA Status : {self.device.type}")

    def setup_datasets(self, args):
        transform = {
            'train': transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomRotation(25),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'valid': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        }
        img_dirs = {'train': '/mnt/HDD/chest-seg/dataset/train_img', 'valid': '/mnt/HDD/chest-seg/dataset/train_img' }
        mask_dir = {'train': '/mnt/HDD/chest-seg/dataset/train_mask', 'valid': '/mnt/HDD/chest-seg/dataset/train_mask'} 

        self.train_loader = DataLoader(
            CustomDataset(image_dir = img_dirs['train'], mask_dir = mask_dir['train'], transform=transform['train'], testing=False,),
            batch_size=args.ts_batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            CustomDataset(image_dir = img_dirs['valid'], mask_dir = mask_dir['valid'], transform=transform['valid'], testing=True, seed = 627), 
            batch_size=args.vs_batch_size, shuffle=True
        )

    def setup_wandb(self, args):
        self.w = args.wandb.strip().lower()
        if args.wandb.strip().lower() == "yes":
            wandb.init(project='Chest-segmentation', entity='dablro1232', notes='baseline', config=args.__dict__)
            wandb.run.name = args.model + f'_{args.version}_{args.training_date}'
            self.run_name = args.model + f'_{args.version}_{args.training_date}'
        else:
            self.run_name = args.model + '_debug'

    def initialize_models(self, args):
        model_loader = load_model.segmentation_models_loader(model_name = args.model)
        self.model = model_loader().to(self.device)
                
        self.setup_optimizers(args)
    
    def setup_train(self, args):
        self.epochs = args.epochs
        self.loss_fn = nn.BCELoss().to(self.device)
        
    def setup_optimizers(self, args):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, betas=(0, 0.9))

        if args.pretrain == 'yes':
            checkpoint_paths = {key: f"/mnt/HDD/chest-seg_models/pretrained_models/{args.pretrained_model}/{key}0000000.pt" for key in ['G', 'D', 'O']}
            self.load_checkpoints(checkpoint_paths)
            print('\033[41m',"#"*30, ' | ', 'Pretrained Setting Complete !!', '\033[0m')
        self.setup_scheduler(args)  # 스케줄러 설정 호출
    
    def setup_scheduler(self, args):
        # ReduceLROnPlateau 스케줄러 설정
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    def load_checkpoints(self, paths):
        self.model_weights = torch.load(paths['G'], map_location=self.device)
        self.model.load_state_dict(self.model_weights)
        self.optim_weights = torch.load(paths['O'], map_location=self.device)
        self.optimizer.load_state_dict(self.optim_weights) #Only use pretrained G optimizer checkpoint 

    def setup_paths(self, args):
        self.save_path = os.path.join(args.save_path, f"{self.run_name}")
        os.makedirs(self.save_path, exist_ok=True)
        save_args(f"{self.save_path}/{self.run_name}.json")

    def fit(self):
        for epoch in tqdm(range(1, self.epochs+1)):
            train_losses, train_accs, train_ious, valid_losses, valid_accs, valid_ious = 0, 0, 0, 0, 0, 0
            # Training phase
            self.model.train()
            for images, masks in self.train_loader:
                self.optimizer.zero_grad()
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                # BCE Logistic Loss
                loss = self.loss_fn(outputs, masks)
                loss.backward()
                self.optimizer.step()

                acc, iou = train_metrics(outputs, masks, threshold=0.5)

                train_losses += loss.cpu().detach().item()
                train_accs += acc
                train_ious += iou
            # Validation phase
            with torch.no_grad():
                self.model.eval()
                for (images, masks, paths) in self.valid_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    # BCE Logistic Loss
                    loss = self.loss_fn(outputs, masks)

                    acc, iou = train_metrics(outputs, masks, threshold=0.5)

                    valid_losses += loss.cpu().detach().item()
                    valid_accs += acc
                    valid_ious += iou
                    
            self.scheduler.step(valid_losses)  # 스케줄러 업데이트 호출
            self.log_metrics(epoch, train_losses, train_accs, train_ious, valid_losses, valid_accs, valid_ious)
            self.visualize(epoch = epoch, image = images[0,0], mask = masks[0,0], output_image = (outputs[0,0] > 0.5).int())
            
            # Early Stopping Check
            if valid_losses < self.best_valid_loss:
                self.best_valid_loss = valid_losses
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.n_epochs_stop:
                self.save_model(epoch, valid_losses/len(self.valid_loader))  # Save the best model
                print(f"\033[41m Early stopping at epoch {epoch}\033[0m")
                print(f"\033[41m Best valid loss: {self.best_valid_loss}\033[0m")
                print(f"\033[41m ACC : {valid_accs / len(self.valid_loader)}\033[0m")
                print(f"\033[41m mIoU: {valid_ious / len(self.valid_loader)}\033[0m")
                break
            
            if epoch % 50 == 0:
                self.save_model(epoch, valid_losses/len(self.valid_loader))

        if self.w == "yes":
            wandb.finish()
        print("Training Complete.")

    def log_metrics(self, epoch, train_losses, train_accs, train_ious, valid_losses, valid_accs, valid_ious):
        if self.w == 'yes':
            wandb.log({
                "train_loss": train_losses / len(self.train_loader),
                "train_acc": train_accs / len(self.train_loader),
                "train_ious": train_ious / len(self.train_loader),
                "valid_loss": valid_losses / len(self.valid_loader),
                "valid_acc": valid_accs / len(self.valid_loader),
                "valid_ious": valid_ious / len(self.valid_loader),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step = epoch)

    def save_model(self, epoch, valid_loss):
        current_path = os.path.join(self.save_path, f"model_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimzier_state_dict": self.optimizer.state_dict(),
            "valid_loss": valid_loss
        }, current_path)
        print(f"Saved model at {current_path}")

    def visualize(self, epoch, image, mask, output_image):
        current_path = os.path.join(self.save_path, f"result_{epoch}.png")

        vis_imgs = [image, mask, output_image]
        vis_labels = ['Input','Mask(GT)','Result']
        plt.figure(figsize = (12, 8))
        for index, plot_img in enumerate(vis_imgs):
            plt.subplot(1,3,index+1)
            plt.imshow(plot_img.cpu().detach().numpy(), cmap = 'gray')
            plt.title(vis_labels[index])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(current_path)
        plt.close()
        
        print(f"Visualize save : {epoch}/{self.epochs}")
        