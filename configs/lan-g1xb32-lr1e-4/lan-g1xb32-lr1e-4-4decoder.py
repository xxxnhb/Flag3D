# train & test
train_batch_size = 32
test_batch_size = 8
num_workers = 8
max_epochs = 30

# language model
bert_type = "roberta-base"
language_path = '/mnt/disk_1/jinpeng/rank/data/FLAG/flag3d_language.pkl'

# co attention
feat_size = 256
co_attention_layer_nhead=8

# Decoder VIT
dim = 256
num_heads = 8
num_layers = 4

# MLP regression
in_channel = 256
out_channel = 1

# learning rate
lr = 1e-4
weight_loss_aqa = 1

# data_path = '/mnt/disk_1/jinpeng/rank/data/FLAG/flag3d_ntu_t.pkl'
data_path = '/mnt/disk_1/jinpeng/rank/data/FLAG/flag3d_ntu_random.pkl'  # random split
voter_number = 10

model = dict(
    type='LanBaseModel',
    decoder=dict(
        type='Decoder',
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers
    ),
    mlp_score=dict(
        type='MLP',
        in_channel=in_channel,
        out_channel=out_channel),
    feat_size=feat_size,
    co_attention_layer_nhead=co_attention_layer_nhead,
    bert_type=bert_type,
    weight_loss_aqa=weight_loss_aqa)

model_wrapper_cfg = dict(
    type='BaseDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

dataset_args = dict(
    path=data_path,
    voter_number=voter_number,
    language_path=language_path)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=False,
    dataset=dict(
        type='LanFlag3D',
        phase='train',
        **dataset_args))

val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LanFlag3D',
        phase='test',
        **dataset_args))

test_dataloader = val_dataloader

metric_prefix = '[ERROR]'
val_evaluator = [dict(type='ErrorMetric', prefix=metric_prefix)]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=lr),
    accumulative_counts=1,
    paramwise_cfg=dict(custom_keys={'cond_net.scene_model': dict(lr_mult=0.1)}))

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best='auto', less_keys=[metric_prefix]),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True,num_digits=6,
                     custom_cfg=[dict(data_src='loss', log_name='loss_epoch', method_name='mean',
                                      window_size='epoch')])

visualizer = dict(type='Visualizer', vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

log_level = 'INFO'
load_from = None
resume = False

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (24 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=train_batch_size)
