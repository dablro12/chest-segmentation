import matplotlib.pyplot as plt 
import matplotlib
import torch 
import os 
import numpy as np 

# matplotlib의 Backend를 TkAgg로 설정
def train_plotting(images, masks):
    plt.figure(dpi =256)
    plt.subplot(121)
    plt.imshow(images[1,0], cmap= 'gray')
    plt.title('image')
    plt.subplot(122)
    plt.imshow(masks[1,0], cmap= 'gray')
    plt.title('mask')
    plt.tight_layout()
    plt.show()
    
def test_plotting(input_image, mask, pred_image, save_path):
    plt.figure(dpi =256)
    plt.subplot(131)
    plt.imshow(input_image.permute(1, 2, 0).cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('Input')
    plt.subplot(132)
    plt.imshow(mask.permute(1, 2, 0).cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('Mark')
    plt.subplot(133)
    plt.imshow(pred_image.permute(1, 2, 0).cpu().detach().numpy(), cmap= 'gray')
    plt.title('Output')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_validation(input_images, masks, output_images, epoch, save_dir):
    plt.figure(dpi=128)
    plt.subplot(231)
    plt.imshow(input_images[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.title("Input")
    plt.subplot(232)
    plt.imshow(masks[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.title("mask")
    plt.subplot(233)
    plt.imshow(output_images[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.title('Output')
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.show()
    plt.close()

def save_model(model, optimizer, epoch, save_dir):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(save_dir, f'epoch_{epoch}.pt'))

def save_loss(metrics, save_dir):
    # loss plot
    plt.figure(dpi=128)
    for key, value in metrics.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    
    np.save(os.path.join(save_dir, 'metrics.npy'),metrics)


def visualize_gui(original_images, masks, results):
    matplotlib.use('TkAgg')

    plt.ion()  # Interactive mode on
    plt.figure(figsize=(12, 4))
    titles = ['Original Image', 'Mask', 'Composite Image']

    # 원본 이미지 표시
    plt.subplot(1, 3, 1)
    plt.title(titles[0])
    plt.imshow(original_images[0].cpu().detach().permute(1, 2, 0))
    plt.title('INPUT')
    plt.axis('off')

    # 마스크 표시
    plt.subplot(1, 3, 2)
    plt.title(titles[1])
    plt.imshow(masks[0].cpu().detach().squeeze(), cmap='gray')
    plt.title('MASK')
    plt.axis('off')

    # 복원된 이미지 표시
    plt.subplot(1, 3, 3)
    plt.title(titles[2])
    plt.imshow(results[0].cpu().detach().permute(1, 2, 0))
    plt.axis('off')
    plt.title('OUTPUT')
    plt.show()
    plt.pause(0.1)  # GUI 창이 업데이트되도록 잠시 대기

    plt.ioff()  # Interactive mode off