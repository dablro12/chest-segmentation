import segmentation_models_pytorch as smp 
def unet_plus_plus():
    model = smp.UnetPlusPlus(
        encoder_name= "resnet34",
        encoder_weights= "imagenet",
        in_channels = 1,
        activation= None,
        classes = 1
        )
    return model 
