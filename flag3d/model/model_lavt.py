import torch
import torch.nn as nn
from collections import OrderedDict
from pytorch_pretrained_bert.modeling import BertModel
import mmengine.dist as dist
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint
from flag3d.dataset.utils import cast_data
from flag3d.model.blocks import TransformerEncoderLayer, SpatialImageLanguageAttention
from transformers import RobertaModel


# use cross attention before decoder vit
@MODELS.register_module()
class LanBaseModel(nn.Module):
    def __init__(self, decoder: dict, mlp_score: dict,
                 bert_type, weight_loss_aqa, feat_size=256, co_attention_layer_nhead=8):
        super(LanBaseModel, self).__init__()
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)
        self.Bert_model = RobertaModel.from_pretrained(bert_type)
        self.feat_size = feat_size
        for param in self.Bert_model.parameters():
            param.requires_grad = False
        self.lang_linear = nn.Linear(self.Bert_model.config.hidden_size, feat_size)

        self.co_attention_layer = TransformerEncoderLayer(
            feat_size,
            co_attention_layer_nhead,
            batch_first=True)
        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target, input_ids, word_mask = data
        #
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = self.lang_linear(language)
        #
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
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2,
                                                                                                                 1)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2,
                                                                                                                 1)

        # co-attention
        video_mask = torch.zeros(video_1.shape[0:2], dtype=torch.bool, device=video_1.device)
        padding_mask_all = torch.cat([video_mask, word_mask], dim=1).type(torch.bool)
        raw_feature1 = torch.cat([video_1, language], dim=1)  # <B, NV+NL, C=256>
        raw_feature2 = torch.cat([video_2, language], dim=1)  # <B, NV+NL, C=256>

        co_attention_output1, attn_score1 = self.co_attention_layer(raw_feature1,
                                                                    src_key_padding_mask=padding_mask_all)
        co_attention_output2, attn_score2 = self.co_attention_layer(raw_feature2,
                                                                    src_key_padding_mask=padding_mask_all)
        video_hidden_states1 = co_attention_output1[:, 0:8, :]  # <B, NV, C=256>
        lang_hidden_states1 = co_attention_output1[:, 8, :].reshape(-1, 1, self.feat_size)  # use cls token feature
        video_hidden_states2 = co_attention_output2[:, 0:8, :]  # <B, NV, C=256>
        lang_hidden_states2 = co_attention_output2[:, 8, :].reshape(-1, 1, self.feat_size)  # use cls token feature

        feature1 = torch.cat((video_hidden_states1, lang_hidden_states1), dim=1)
        feature2 = torch.cat((video_hidden_states2, lang_hidden_states2), dim=1)

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(feature1, feature2)
        decoder_video_21_map = self.Decoder_vit(feature2, feature1)

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
        sample, examples, input_ids, word_mask = data
        # language process
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = self.lang_linear(language)
        # video_1 process
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]

        # video_1 feature
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
            permute(0, 2, 1)

        # video_1 and vide_2 have the same shape, so the mask is same
        video_mask = torch.zeros(video_1.shape[0:2], dtype=torch.bool, device=video_1.device)
        padding_mask_all = torch.cat([video_mask, word_mask], dim=1).type(torch.bool)
        raw_feature1 = torch.cat([video_1, language], dim=1)  # <B, NV+NL, C=256>

        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            # video_2 feature
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
                permute(0, 2, 1)
            raw_feature2 = torch.cat([video_2, language], dim=1)  # <B, NV+NL, C=256>

            # co_attention
            co_attention_output1, attn_score1 = self.co_attention_layer(raw_feature1,
                                                                        src_key_padding_mask=padding_mask_all)
            co_attention_output2, attn_score2 = self.co_attention_layer(raw_feature2,
                                                                        src_key_padding_mask=padding_mask_all)
            video_hidden_states1 = co_attention_output1[:, 0:8, :]  # <B, NV, C=256>
            lang_hidden_states1 = co_attention_output1[:, 8, :].reshape(-1, 1, self.feat_size)  # use cls token feature
            video_hidden_states2 = co_attention_output2[:, 0:8, :]  # <B, NV, C=256>
            lang_hidden_states2 = co_attention_output2[:, 8, :].reshape(-1, 1, self.feat_size)  # use cls token feature

            feature1 = torch.cat((video_hidden_states1, lang_hidden_states1), dim=1)
            feature2 = torch.cat((video_hidden_states2, lang_hidden_states2), dim=1)

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(feature1, feature2)
            decoder_video_21_map = self.Decoder_vit(feature2, feature1)

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


# only concate
@MODELS.register_module()
class LanConcateModel(nn.Module):
    def __init__(self, decoder: dict, mlp_score: dict,
                 bert_type, weight_loss_aqa, cls, feat_size=256):
        super(LanConcateModel, self).__init__()
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)
        self.Bert_model = RobertaModel.from_pretrained(bert_type)
        self.feat_size = feat_size
        self.cls = cls
        for param in self.Bert_model.parameters():
            param.requires_grad = False
        self.lang_linear = nn.Linear(self.Bert_model.config.hidden_size, feat_size)

        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target, input_ids, word_mask = data
        #
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = self.lang_linear(language)
        if self.cls:
            language = language[:, 0, :].reshape(-1, 1, self.feat_size)  # cls token feature
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
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2,
                                                                                                                 1)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2,
                                                                                                                 1)

        # concat
        feature1 = torch.cat([video_1, language], dim=1)  # <B, NV+NL, C=256>
        feature2 = torch.cat([video_2, language], dim=1)  # <B, NV+NL, C=256>

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(feature1, feature2)
        decoder_video_21_map = self.Decoder_vit(feature2, feature1)

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
        sample, examples, input_ids, word_mask = data
        # language process
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = self.lang_linear(language)
        if self.cls:
            language = language[:, 0, :].reshape(-1, 1, self.feat_size)  # cls token feature
        # video_1 process
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]

        # video_1 feature
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
            permute(0, 2, 1)

        feature1 = torch.cat([video_1, language], dim=1)  # <B, NV+NL, C=256>

        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            # video_2 feature
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
                permute(0, 2, 1)
            feature2 = torch.cat([video_2, language], dim=1)  # <B, NV+NL, C=256>

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(feature1, feature2)
            decoder_video_21_map = self.Decoder_vit(feature2, feature1)

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


# cross attention lost language
@MODELS.register_module()
class LanReferModel(nn.Module):
    def __init__(self, decoder: dict, mlp_score: dict,
                 bert_type, weight_loss_aqa, feat_size=256, co_attention_layer_nhead=8):
        super(LanReferModel, self).__init__()
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)
        self.Bert_model = RobertaModel.from_pretrained(bert_type)
        self.feat_size = feat_size
        for param in self.Bert_model.parameters():
            param.requires_grad = False
        self.lang_linear = nn.Linear(self.Bert_model.config.hidden_size, feat_size)

        self.co_attention_layer = TransformerEncoderLayer(
            feat_size,
            co_attention_layer_nhead,
            batch_first=True)
        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target, input_ids, word_mask = data
        #
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = self.lang_linear(language)
        #
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
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2,
                                                                                                                 1)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2,
                                                                                                                 1)

        # co-attention
        video_mask = torch.zeros(video_1.shape[0:2], dtype=torch.bool, device=video_1.device)
        padding_mask_all = torch.cat([video_mask, word_mask], dim=1).type(torch.bool)
        raw_feature1 = torch.cat([video_1, language], dim=1)  # <B, NV+NL, C=256>
        raw_feature2 = torch.cat([video_2, language], dim=1)  # <B, NV+NL, C=256>

        co_attention_output1, attn_score1 = self.co_attention_layer(raw_feature1,
                                                                    src_key_padding_mask=padding_mask_all)
        co_attention_output2, attn_score2 = self.co_attention_layer(raw_feature2,
                                                                    src_key_padding_mask=padding_mask_all)
        video_hidden_states1 = co_attention_output1[:, 0:8, :]  # <B, NV, C=256>
        video_hidden_states2 = co_attention_output2[:, 0:8, :]  # <B, NV, C=256>

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(video_hidden_states1, video_hidden_states2)
        decoder_video_21_map = self.Decoder_vit(video_hidden_states2, video_hidden_states1)

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
        sample, examples, input_ids, word_mask = data
        # language process
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = self.lang_linear(language)
        # video_1 process
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]

        # video_1 feature
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
            permute(0, 2, 1)

        # video_1 and vide_2 have the same shape, so the mask is same
        video_mask = torch.zeros(video_1.shape[0:2], dtype=torch.bool, device=video_1.device)
        padding_mask_all = torch.cat([video_mask, word_mask], dim=1).type(torch.bool)
        raw_feature1 = torch.cat([video_1, language], dim=1)  # <B, NV+NL, C=256>

        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            # video_2 feature
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1). \
                permute(0, 2, 1)
            raw_feature2 = torch.cat([video_2, language], dim=1)  # <B, NV+NL, C=256>

            # co_attention
            co_attention_output1, attn_score1 = self.co_attention_layer(raw_feature1,
                                                                        src_key_padding_mask=padding_mask_all)
            co_attention_output2, attn_score2 = self.co_attention_layer(raw_feature2,
                                                                        src_key_padding_mask=padding_mask_all)
            video_hidden_states1 = co_attention_output1[:, 0:8, :]  # <B, NV, C=256>
            video_hidden_states2 = co_attention_output2[:, 0:8, :]  # <B, NV, C=256>

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(video_hidden_states1, video_hidden_states2)
            decoder_video_21_map = self.Decoder_vit(video_hidden_states2, video_hidden_states1)

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


@MODELS.register_module()
class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=1, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                         )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm  # B(B, 256, 8)


# use PWAM model to fusion feature
@MODELS.register_module()
class PWAMModel(nn.Module):
    def __init__(self, pwam: dict, decoder: dict,
                 mlp_score: dict, bert_type, cls, weight_loss_aqa,
                 feat_size=256):
        super(PWAMModel, self).__init__()
        self.PWAM = MODELS.build(pwam)
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)
        self.Bert_model = RobertaModel.from_pretrained(bert_type)
        self.feat_size = feat_size
        self.cls = cls
        for param in self.Bert_model.parameters():
            param.requires_grad = False

        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target, input_ids, word_mask = data
        #
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = language.permute(0, 2, 1)  # to make Conv1d happy (B£¬ L£¬ 768) -> (B, 768, L)
        if self.cls:
            language = language[:, :, 0].reshape(-1, self.Bert_model.config.hidden_size,
                                                 1)  # cls token feature (B, 768, 1)
            word_mask = word_mask[:, 0].reshape(-1, 1, 1)
        else:
            word_mask = word_mask.unsqueeze(dim=-1)
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
        # video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2, 1)  # (B, 256, 8)
        # video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1).permute(0, 2, 1)
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)  # (B, 256, 8)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)

        # PWAM(B, C, H*W)
        feature1 = self.PWAM(video_1, language, word_mask).permute(0, 2, 1)  # video1(B, 256, 8) language(B, 768, 1)
        feature2 = self.PWAM(video_2, language, word_mask).permute(0, 2, 1)

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(feature1, feature2)
        decoder_video_21_map = self.Decoder_vit(feature2, feature1)

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
        sample, examples, input_ids, word_mask = data
        # language process
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        language = language.permute(0, 2, 1)  # to make Conv1d happy (B£¬ 768£¬ L)
        if self.cls:
            language = language[:, :, 0].reshape(-1, self.Bert_model.config.hidden_size,
                                                 1)  # cls token feature (B, 768, 1)
            word_mask = word_mask[:, 0].reshape(-1, 1, 1)
        else:
            word_mask = word_mask.unsqueeze(dim=-1)
        # video_1 process
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]

        # video_1 feature
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)
        feature1 = self.PWAM(video_1, language, word_mask).permute(0, 2, 1)

        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            # video_2 feature
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)
            feature2 = self.PWAM(video_2, language, word_mask).permute(0, 2, 1)

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(feature1, feature2)
            decoder_video_21_map = self.Decoder_vit(feature2, feature1)

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


# language pathway + PWAM
@MODELS.register_module()
class PWAMAddModel(nn.Module):
    def __init__(self, pwam: dict, decoder: dict,
                 mlp_score: dict, bert_type, cls, projector, weight_loss_aqa, dim,
                 feat_size=256):
        super(PWAMAddModel, self).__init__()
        self.PWAM = MODELS.build(pwam)
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)
        self.Bert_model = RobertaModel.from_pretrained(bert_type)
        self.text_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.LayerNorm(256, eps=1e-12),
            nn.Dropout(0.1)
        )
        self.feat_size = feat_size
        self.cls = cls
        self.projector = projector
        if self.projector:
            self.language_size = feat_size
        else:
            self.language_size = self.Bert_model.config.hidden_size
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        for param in self.Bert_model.parameters():
            param.requires_grad = False

        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target, input_ids, word_mask = data
        #
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        if self.projector:
            language = self.text_projector(language)     # 768->256
        language = language.permute(0, 2, 1)  # to make Conv1d happy (B£¬ L£¬ 768) -> (B, 768, L)
        if self.cls:
            language = language[:, :, 0].reshape(-1, self.language_size,
                                                 1)  # cls token feature (B, 768, 1)
            word_mask = word_mask[:, 0].reshape(-1, 1, 1)
        else:
            word_mask = word_mask.unsqueeze(dim=-1)
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
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)  # (B, 256, 8)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)

        # PWAM(B, C, H*W)
        pwam_feature1 = self.PWAM(video_1, language, word_mask)  # video1(B, 256, 8) language(B, 768, 1) (B, 1, 1)
        pwam_feature2 = self.PWAM(video_2, language, word_mask)

        # language pathway
        feature1 = video_1 + self.res_gate(pwam_feature1) * pwam_feature1
        feature2 = video_2 + self.res_gate(pwam_feature2) * pwam_feature2

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(feature1.permute(0, 2, 1), feature2.permute(0, 2, 1))
        decoder_video_21_map = self.Decoder_vit(feature2.permute(0, 2, 1), feature1.permute(0, 2, 1))

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
        sample, examples, input_ids, word_mask = data
        # language process
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        if self.projector:
            language = self.text_projector(language)     # 768->256
        language = language.permute(0, 2, 1)  # to make Conv1d happy (B£¬ 768£¬ L)
        if self.cls:
            language = language[:, :, 0].reshape(-1, self.language_size,
                                                 1)  # cls token feature (B, 768, 1)
            word_mask = word_mask[:, 0].reshape(-1, 1, 1)
        else:
            word_mask = word_mask.unsqueeze(dim=-1)
        # video_1 process
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]

        # video_1 feature
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)
        pwam_feature1 = self.PWAM(video_1, language, word_mask)
        feature1 = video_1 + self.res_gate(pwam_feature1) * pwam_feature1

        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            # video_2 feature
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)
            pwam_feature2 = self.PWAM(video_2, language, word_mask)
            feature2 = video_2 + self.res_gate(pwam_feature2) * pwam_feature2

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(feature1.permute(0, 2, 1), feature2.permute(0, 2, 1))
            decoder_video_21_map = self.Decoder_vit(feature2.permute(0, 2, 1), feature1.permute(0, 2, 1))

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


# language pathway + PWAM
@MODELS.register_module()
class HierarchyModel(nn.Module):
    def __init__(self, pwam: dict, decoder: dict,
                 mlp_score: dict, bert_type, cls, projector, weight_loss_aqa, dim,
                 feat_size=256):
        super(HierarchyModel, self).__init__()
        self.PWAM = MODELS.build(pwam)
        self.Decoder_vit = MODELS.build(decoder)
        self.Regressor_delta = MODELS.build(mlp_score)
        self.Bert_model = RobertaModel.from_pretrained(bert_type)
        self.text_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256, eps=1e-12),
            nn.Dropout(0.1)
        )
        self.feat_size = feat_size
        self.cls = cls
        self.projector = projector
        if self.projector:
            self.language_size = feat_size
        else:
            self.language_size = self.Bert_model.config.hidden_size
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        for param in self.Bert_model.parameters():
            param.requires_grad = False

        self.weight_loss_aqa = weight_loss_aqa

    def forward(self, data, mode='train'):
        data = cast_data(data)
        sample, target, input_ids, word_mask = data
        #
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        if self.projector:
            language = self.text_projector(language)     # 768->256
        language = language.permute(0, 2, 1)  # to make Conv1d happy (B£¬ L£¬ 768) -> (B, 768, L)
        if self.cls:
            language = language[:, :, 0].reshape(-1, self.language_size,
                                                 1)  # cls token feature (B, 768, 1)
            word_mask = word_mask[:, 0].reshape(-1, 1, 1)
        else:
            word_mask = word_mask.unsqueeze(dim=-1)
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
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)  # (B, 256, 8)
        video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)

        # PWAM(B, C, H*W)
        pwam_feature1 = self.PWAM(video_1, language, word_mask)  # video1(B, 256, 8) language(B, 768, 1) (B, 1, 1)
        pwam_feature2 = self.PWAM(video_2, language, word_mask)

        # language pathway
        feature1 = video_1 + self.res_gate(pwam_feature1) * pwam_feature1
        feature2 = video_2 + self.res_gate(pwam_feature2) * pwam_feature2

        # VIT decoder
        decoder_video_12_map = self.Decoder_vit(feature1.permute(0, 2, 1), feature2.permute(0, 2, 1), pwam_feature1.permute(0, 2, 1))
        decoder_video_21_map = self.Decoder_vit(feature2.permute(0, 2, 1), feature1.permute(0, 2, 1), pwam_feature2.permute(0, 2, 1))

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
        sample, examples, input_ids, word_mask = data
        # language process
        input_ids = torch.stack(input_ids)
        word_mask = torch.stack(word_mask)
        language = self.Bert_model(input_ids, attention_mask=word_mask)
        language = language.last_hidden_state
        if self.projector:
            language = self.text_projector(language)     # 768->256
        language = language.permute(0, 2, 1)  # to make Conv1d happy (B£¬ 768£¬ L)
        if self.cls:
            language = language[:, :, 0].reshape(-1, self.language_size,
                                                 1)  # cls token feature (B, 768, 1)
            word_mask = word_mask[:, 0].reshape(-1, 1, 1)
        else:
            word_mask = word_mask.unsqueeze(dim=-1)
        # video_1 process
        video_1 = sample['video']
        label_1 = sample['final_score']
        video_1 = torch.stack(video_1)
        video_2_list = [item['video'] for item in examples]
        label_2_list = [item['final_score'] for item in examples]

        # video_1 feature
        score = 0
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]
        video_1 = torch.cat([video_1[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)
        pwam_feature1 = self.PWAM(video_1, language, word_mask)
        feature1 = video_1 + self.res_gate(pwam_feature1) * pwam_feature1

        # skeleton feature
        for video_2, label_2_score in zip(video_2_list, label_2_list):
            # video_2 feature
            video_2 = torch.stack(video_2)
            label_2_score = torch.stack(label_2_score)
            video_2 = torch.cat([video_2[:, :, i: i + 8].unsqueeze(-1) for i in start_idx], dim=-1).mean(-1)
            pwam_feature2 = self.PWAM(video_2, language, word_mask)
            feature2 = video_2 + self.res_gate(pwam_feature2) * pwam_feature2

            # VIT decoder
            decoder_video_12_map = self.Decoder_vit(feature1.permute(0, 2, 1), feature2.permute(0, 2, 1), pwam_feature1.permute(0, 2, 1))
            decoder_video_21_map = self.Decoder_vit(feature2.permute(0, 2, 1), feature1.permute(0, 2, 1), pwam_feature2.permute(0, 2, 1))

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