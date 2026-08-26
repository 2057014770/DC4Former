_base_ = ['./segformer_base_floodnet-512x512.py']

custom_imports = dict(imports=['dc4former'], allow_failed_imports=False)

# ===================== 仅修改：预训练权重路径（适配 FloodNet 环境） =====================
pretrained = 'pretrained/lrformer_small.pth'
depths = [3, 3, 12, 3]

# ===================== 模型结构：完全复刻 RescueNet 配置（无任何修改） =====================
model = dict(
    backbone=dict(
        type='DC4Former',
        _delete_=True,

        # 完全和 RescueNet 一致
        q_conv=False,
        use_ls=True,
        extra_act=False,
        q_pooled_sizes=[16, 16, 16, 16],  # 用 RescueNet 参数，删除原 FloodNet 的 q_pool_ratios
        embed_dims=[64, 128, 320, 512],
        depths=depths,
        drop_path_rate=0.15,  # RescueNet: 0.15（弃用 FloodNet 原 0.3）

        # TGRS Strong-C4 + ESAF 融合模块：完全保留 RescueNet 配置
        c4_conv=True,
        c4_attention=True,
        c4_downsample=True,
        fusion_type='cl_esaf',
        fusion_alpha_max=0.25,
        fusion_alpha_init=-2.0,
        fusion_temp_init=2.0,

        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512])
)

# ===================== 训练配置：完全对齐 RescueNet =====================
train_dataloader = dict(batch_size=16)

# 优化器：和 RescueNet 完全一致
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999)
)

backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)

# 参数分组策略：1:1 复刻 RescueNet
custom_keys = {
    'backbone': dict(lr_mult=0.2, decay_mult=1.0),
    'fusion_scorer': dict(lr_mult=1.0, decay_mult=1.0),
    'fusion_log_temp': dict(lr_mult=2.0, decay_mult=0.0),
    'fusion_alpha_logit': dict(lr_mult=2.0, decay_mult=0.0),
    'd_convs': dict(lr_mult=0.5, decay_mult=1.0),
    'decode_head': dict(lr_mult=1.0, decay_mult=1.0),
    'norm': dict(decay_mult=0.0),
    'bn': dict(decay_mult=0.0),
    'bias': dict(decay_mult=0.0),
}

custom_keys.update({
    f'backbone.block{stage_id + 1}.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id + 1}.norm': backbone_norm_multi
    for stage_id in range(len(depths))
})
custom_keys.update({
    f'backbone.block{stage_id + 1}.{block_id}.attn.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})

# 优化器包装器：AMP 混合精度 + 梯度裁剪 完全对齐 RescueNet
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    accumulative_counts=1,
    optimizer=optimizer,
    clip_grad=dict(max_norm=1.0, norm_type=2),  # RescueNet: 1.0（弃用 FloodNet 原 0.1）
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)

# 学习率调度：两段式（预热+PolyLR）完全对齐 RescueNet
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=1500, end=75000, by_epoch=False)
]

# 训练循环配置：论文中的 FloodNet 训练设置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=75000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 环境配置：完全一致
env_cfg = dict(cudnn_benchmark=False)
randomness = dict(seed=0, deterministic=False)
