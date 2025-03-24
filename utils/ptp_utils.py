import abc
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from collections import defaultdict
from typing import List


class EmptyControl:
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, head: int):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            # attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, head // 2)
            attn = self.forward(attn, is_cross, place_in_unet, head)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def skip(self):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            'down_cross': [],
            'mid_cross': [],
            'up_cross': [],
            'down_self': [],
            'mid_self': [],
            'up_self': []
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str, head: int):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append((head, attn))
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i][1] += self.step_store[key][i][1]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [(head, item / self.cur_step) for head, item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.is_able = False

    def __enter__(self):
        self.is_able = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.is_able = False


def aggregate_attention(attention_store: AttentionStore, spatial_res: int, temporal_res: int,
                        from_where: List[str], bsz: int, f: int):
    out_spatial = []
    out_temporal = []
    attention_maps = attention_store.get_average_attention()
    spatial_pixels = spatial_res**2
    temporal_pixels = temporal_res**2
    for location in from_where:
        for head, item in attention_maps[f"{location}_cross"]:
            if item.shape[1] == spatial_pixels:
                cross_maps = item.reshape(bsz, f, -1, spatial_res, spatial_res, item.shape[-1])
                out_spatial.append(cross_maps)
        for head, item in attention_maps[f"{location}_self"]:
            if item.shape[0] == bsz * temporal_pixels * head and item.shape[1] == f:
                self_maps = item.reshape((bsz, temporal_pixels, head) + item.shape[-2:])
                out_temporal.append(self_maps)
    out_spatial = torch.cat(out_spatial, dim=2).mean(dim=2)
    out_temporal = torch.cat(out_temporal, dim=2).mean(dim=2)
    return out_spatial, out_temporal
