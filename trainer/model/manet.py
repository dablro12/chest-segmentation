import segmentation_models_pytorch as smp 
def manet():
    model = smp.MAnet(
        encoder_name= "resnet34",
        encoder_weights= "imagenet",
        in_channels = 1,
        classes = 1
        )
    return model
