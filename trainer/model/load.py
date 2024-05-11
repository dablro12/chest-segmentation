
import torch

class load_model:
    def __init__(self, model_name, path, device):
        self.model_name = model_name
        self.path = path
        self.device = device
    
    def load(self, model_name):
        if model_name == 'aot-gan':
            model = self.aot_gan().to(self.device)
            print(f"model name : {model_name}")
        else:
            raise ValueError('Model not found')        
        
        model = self.load_weight(model = model)
        return model
    
    def load_weight(self, model):
        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['netG'])
        print('#'*30,'Load Pretrained Weight','#'*30)
        return model 
    
    def aot_gan(self):
        from model.aotgan import InpaintGenerator, Discriminator
        
        model = InpaintGenerator()
        return model
    
    def get(self):
        model = self.load(model_name = self.model_name)
        return model
    
        

        
        
        
            