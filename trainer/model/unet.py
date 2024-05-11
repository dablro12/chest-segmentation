import torch 
def U_Net():
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=1,
        out_channels=1,
        init_features=32,
        pretrained=True
    )
    return model
