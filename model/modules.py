import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip
import yaml
# from group_modules import *
# from cbam import CBAM
from torchsummary import summary
from einops import rearrange
from decord import VideoReader, cpu
import numpy as np

from transformers import VideoMAEFeatureExtractor, VideoMAEModel
from huggingface_hub import hf_hub_download
from easydict import EasyDict
from model.slowfast.models import build_model
from model.slowfast.config.defaults import get_cfg
import model.slowfast.utils.checkpoint as cu
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class FC(nn.Module):
    def __init__(self, in_size, out_size, pdrop=0., use_gelu=True):
        super(FC, self).__init__()
        self.pdrop = pdrop
        self.use_gelu = use_gelu

        self.linear = nn.Linear(in_size, out_size)

        if use_gelu:
            #self.relu = nn.Relu(inplace=True)
            self.gelu = nn.GELU()

        if pdrop > 0:
            self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.linear(x)

        if self.use_gelu:
            #x = self.relu(x)
            x = self.gelu(x)

        if self.pdrop > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, pdrop=0., use_gelu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, pdrop=pdrop, use_gelu=use_gelu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size=512, flat_glimpses=1, flat_out_size=1024, pdrop=0.01):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            pdrop=pdrop,
            use_gelu=True
        )
        self.flat_glimpses = flat_glimpses

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x):
        att = self.mlp(x)
        
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class SlowFast(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_file(config['config_file'])
        self.config = cfg
        self.config.NUM_GPUS = min(torch.cuda.device_count(), self.config.NUM_GPUS)
        self.model = build_model(self.config)
        
    def forward(self, x):
        
        return self.model(x)
        
class VideoMae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        # self.model.embeddings.position_embeddings = get_sinusoid_encoding_table(14*14*5, self.model.config.hidden_size)
        if not config['use_position_embedding']:
            print('not use position embedding')
            self.model.embeddings.position_embeddings = torch.zeros_like(self.model.embeddings.position_embeddings, requires_grad=False)
        
        self.attn_flat = AttFlat(hidden_size=768)
        
    # B, 16, 3, 224, 224
    def forward(self, x):
        B = x.shape[0]
        num_frames = x.shape[1]
        H = x.shape[-2]//16
        W = x.shape[-1]//16
        
        outputs = self.model(x)
        x = outputs.last_hidden_state # B, 8*14*14, 768
        print(f"x:{x.shape}")
        x = self.attn_flat(x)
        
        return x # B, 768

class Classifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_channels*2, in_channels*2),
                                    nn.ReLU(),
                                    nn.Linear(in_channels*2, 2))
    
    def forward(self, x):
        
        return self.classifier(x)

if __name__ == '__main__':
    # model = FrameEncoder()
    clip_model, _ = clip.load("RN50")
    img = clip_model.encode_image(torch.randn(1, 3, 224, 224).cuda())
    print(img.shape)
    # model = ImageNetEncoder()
    # print(model)