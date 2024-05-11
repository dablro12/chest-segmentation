import sys 
import torch.nn as nn

from model import models
from model import unet, unet_plus_plus, manet, swinunet

class segmentation_models_loader:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self):
        return self.load_model()

    def load_model(self):
        if self.model_name == 'unet':
            model = unet.U_Net()
            print("Model: U-Net loaded successfully!! | pretrained : brain MRI dataset")
        elif self.model_name == 'r2unet':
            model = models.R2U_Net()
            print("Model: R2U_Net loaded successfully!! | pretrained : False")
        elif self.model_name == 'attunet':
            model = models.AttU_Net()
            print("Model: AttU_Net loaded successfully!! | pretrained : False")
        elif self.model_name == 'r2attunet':
            model = models.R2AttU_Net()
            print("Model: R2AttU_Net loaded successfully!! | pretrained : False")
        elif self.model_name == 'unet_plus_plus':
            model = unet_plus_plus.unet_plus_plus()
            print("Model: Unet++ loaded successfully!! | pretrained : imagenet")
        elif self.model_name == 'manet':
            model = manet.manet()
            print("Model: MAnet loaded successfully!! | pretrained : imagenet")
        elif self.model_name == 'monai_swinunet':
            model = swinunet.monai_swinunet()
            print("Model: SwinUNET loaded successfully!! | pretrained : None")
        else:
            raise ValueError('Model name is not valid')
        return model