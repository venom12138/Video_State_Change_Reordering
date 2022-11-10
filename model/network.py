import torch
import torch.nn as nn
from copy import deepcopy
from model.modules import *
# from aggregate import aggregate
# from modules import *
# from memory_util import *
from torchsummary import summary
import clip
import yaml
import pickle
from collections import OrderedDict
from model.slowfast.utils.c2_model_loading import get_name_convert_func

class FrameReorderNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['repr_type'] == 'VideoMae':
            self.Encoder = VideoMae(config) # 输出是1024维
            self.Classifier = Classifier(1024)
        
        elif self.config['repr_type'] == 'SlowFast':
            self.Encoder = SlowFast(config) # 输出是2048维 
            # print(f"SlowFast model:{self.Encoder}")
            self.Classifier = Classifier(2304)
        else:
            raise NotImplementedError
        
        if config['load_network'] != '':
            model_path = config['load_network']
            print('Loading model from {}'.format(model_path))
            
            if config['load_network'].endswith('.pyth'):
                model_weights = torch.load(model_path)
                model_weights = model_weights['model_state']
                for key in list(model_weights.keys()):
                    if 'head.projection_verb' in key or 'head.projection_noun' in key:
                        model_weights.pop(key)
                
            elif config['load_network'].endswith('.pkl'):
                with open(config['load_network'], "rb") as f:
                    caffe2_checkpoint = pickle.load(f, encoding="latin1")
                state_dict = OrderedDict()
                name_convert_func = get_name_convert_func()
                for key in caffe2_checkpoint["blobs"].keys():
                    converted_key = name_convert_func(key)
                    
                    if converted_key in self.Encoder.model.state_dict():
                        if caffe2_checkpoint["blobs"][key].shape == tuple(
                            self.Encoder.model.state_dict()[converted_key].shape
                        ):
                            state_dict[converted_key] = torch.tensor(
                                caffe2_checkpoint["blobs"][key]
                            ).clone()
                            
                        else:
                            print(f'warning!!!')
                    else:
                        
                        if not any(
                            prefix in key for prefix in ["momentum", "lr", "model_iter"]
                        ):
                            print(f"converted_key: {converted_key}")
                            # print(f'big warning: {key}')
                model_weights = state_dict
            
            else:
                
                model_weights = torch.load(model_path)
            
            self.load_model(model_weights)

        if self.config['freeze']:
            print('Freezing the encoder')
            for param in self.Encoder.parameters():
                param.requires_grad = False
    
    # x: B, 5, nf, 3, H, W
    def encode(self, x):
        if self.config['repr_type'] == 'VideoMae':
            B = x.shape[0]
            assert x.shape[1] == 5
            x = x.view(B*5, *x.shape[2:])
            x = self.Encoder(x)
        elif self.config['repr_type'] == 'SlowFast':
            assert len(x) == 2
            B = x[0].shape[0]
            assert x[0].shape[1] == 5
            # B*5, nf. 3, H, W
            x = [x[0].view(B*5, *x[0].shape[2:]), x[1].view(B*5, *x[1].shape[2:])]
            # B*5, 3, nf, H, W
            x = [x[0].permute(0, 2, 1, 3, 4), x[1].permute(0, 2, 1, 3, 4)]
            x = self.Encoder(x)
        x = x.view(B, 5, *x.shape[1:])
        print(f"x output:{x.shape}")
        return x.to(self.Classifier.classifier[0].weight.dtype)  # [B, numframes, 768]
    
    # x: [B,1024*2/2048*2] 
    def classify(self, x):
        return self.Classifier(x) # B,2
    
    # def forward(self, x1, x2):
    #     x1 = self.Encoder(x1) # [B, 2048]
    #     x2 = self.Encoder(x2) # [B, 2048]
    #     x = torch.cat([x1, x2], dim=1) # [B, 2048*2]
    #     x = self.CLassifier(x)
        
    #     return x
    
    def load_model(self, src_dict):
        assert self.config['repr_type'] in ['SlowFast', 'VideoMae']
        if self.config['repr_type'] == 'SlowFast':
            
            self.Encoder.model.load_state_dict(src_dict)
        
        elif self.config['repr_type'] == 'VideoMae':
            raise NotImplementedError