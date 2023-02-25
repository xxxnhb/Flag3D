import copy
import random
from typing import Tuple, Callable
import smplx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pytorch_pretrained_bert.modeling import BertModel
import mmengine.dist as dist
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint
from flag3d.dataset.utils import cast_data
from flag3d.model.blocks import DecoderBlock


# from flag3d.dataset import cast_data
# from flag3d.model import (PointTransformerEnc, ResBlock,
#                        TransformerEncoderLayer, PositionalEncoding, NearestEmbed)
# from flag3d.utils import GeometryTransformer, marker_indic, smplx_signed_distance


def parse_losses(losses):
    loss = losses['loss']
    for loss_name, loss_value in losses.items():
        loss_value = loss_value.data.clone()
        dist.all_reduce(loss_value, 'mean')
        losses[loss_name] = loss_value.item()
    return loss, losses


@MODELS.register_module()
class Decoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super(Decoder, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(DecoderBlock(dim, num_heads))
        self.model = nn.ModuleList(model_list)

    def forward(self, q, v):
        for _layer in self.model:
            q = _layer(q, v)
        return q


@MODELS.register_module()
class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Linear(in_channel, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.layer3(x)
        return output


@MODELS.register_module()
class BaseModel(nn.Module):
    def __init__(self, decoder: dict, mlp_score: dict,
                 weight_loss_aqa):
        super(BaseModel, self).__init__()
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)

        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target = data
        video_1 = sample['video']
        video_2 = target['video']
        label_1_score = sample['final_score']
        label_2_score = target['final_score']
        video_1 = torch.stack(video_1)
        video_2 = torch.stack(video_2)
        label_1_score = torch.stack(label_1_score)
        label_2_score = torch.stack(label_2_score)
        # skeleton feature
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2, 1)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2, 1)

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(video_1, video_2)
        decoder_video_21_map = self.Decoder_vit(video_2, video_1)

        # Fine-grained Contrastive Regression
        decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = self.Regressor_delta(decoder_12_21)
        delta = delta.mean(1).squeeze()
        calc_func = self._calc_loss

        score = delta[:delta.shape[0] // 2].detach() + label_2_score
        loss_metrics = calc_func(
            delta, label_1_score, label_2_score
        )
        return score, loss_metrics

    def forward_test(self, data, mode='test'):
        data = cast_data(data)
        sample, examples = data
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
            permute(0, 2, 1)
        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).\
                permute(0, 2, 1)

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(video_1, video_2)
            decoder_video_21_map = self.Decoder_vit(video_2, video_1)

            # Fine-grained Contrastive Regression
            decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
            delta = self.Regressor_delta(decoder_12_21)
            delta = delta.mean(1).squeeze()
            score += (delta[:delta.shape[0] // 2].detach().T + label_2_score)

        return score / len(video_2_list), label_1

    def _calc_loss(self, delta, label_1_score, label_2_score):
        loss_func = nn.MSELoss()
        loss_aqa = loss_func(delta[:delta.shape[0] // 2], (label_1_score - label_2_score)) + loss_func(
            delta[delta.shape[0] // 2:], (label_2_score - label_1_score))

        losses = [
            (f'aqa_loss*[{self.weight_loss_aqa}]', loss_aqa * self.weight_loss_aqa),
        ]

        loss = sum(value for key, value in losses)
        losses.insert(0, ['loss', loss])
        losses = OrderedDict(losses)
        return losses

    def train_step(self, data, optim_wrapper):
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            score, losses = self(data, mode='train')
        loss, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(loss)
        return score, log_vars

    def val_step(self, data):
        score, label = self.forward_test(data, mode='val')
        return score, label

    def test_step(self, data):
        score, label = self.forward_test(data, mode='val')
        return score, label

    def parse_losses(self, losses):
        loss = losses['loss']
        for loss_name, loss_value in losses.items():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value, 'mean')
            losses[loss_name] = loss_value.item()
        return loss, losses