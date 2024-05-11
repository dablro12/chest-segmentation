import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import Test_CustomDataset
from model.aotgan import InpaintGenerator
import utils

class InpaintingService:
    def __init__(self, img_dir, mask_dir, model_path, batch_size=16):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.netG = InpaintGenerator().to(self.device)
        self.load_model()

        # MKL library 환경 변수 설정
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        # 데이터 로더 설정
        self.test_loader = self.setup_dataloader()

    def load_model(self):
        model_weights = torch.load(self.model_path, map_location=self.device)
        self.netG.load_state_dict(model_weights)
        self.netG.eval()

    def setup_dataloader(self):
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        test_dataset = Test_CustomDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            transform=test_transform,
            mask_transform=None,
        )

        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader

    def infer(self):
        with torch.no_grad():
            for images, masks, image_paths in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                # 마스크된 부분의 이미지 예측
                pred_images = self.netG(images, masks)

                # 원본 이미지와 예측 이미지 결합
                comp_images = images.clone()
                comp_images[masks.repeat(1, 3, 1, 1) != 0] = pred_images[masks.repeat(1, 3, 1, 1) != 0]

                # 결과를 gui로 보여주기 
                utils.visualize_gui(images[0], masks[0], comp_images[0])
# 사용 예제
if __name__ == "__main__":
    img_dir = '/mnt/HDD/octc/mask_abstract/test'
    mask_dir = '/mnt/HDD/octc/mask_abstract/mask'
    model_path = '/mnt/HDD/oci_models/aotgan/OCI-GAN_v1_240506/model_10.pt'

    inpainting_service = InpaintingService(img_dir, mask_dir, model_path)
    inpainting_service.infer()
