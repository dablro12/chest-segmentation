from monai.networks.nets import swin_unetr
from einops import rearrange
from typing import Any

def monai_swinunet():
    model = swin_unetr.SwinUNETR(
        img_size = (256,256),
        in_channels = 1,
        out_channels = 1,
        use_checkpoint = True,
        spatial_dims = 2,
    )

    return model