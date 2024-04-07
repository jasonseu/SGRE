import torch
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_L_16_Weights


weights = ViT_B_16_Weights.verify(ViT_B_16_Weights.IMAGENET1K_V1)
state_dict = weights.get_state_dict(progress=False)
timm_state_dict = {}
for name, param in state_dict.items():
    if name == 'class_token':
        name = 'cls_token'
    if name == 'encoder.pos_embedding':
        name = 'pos_embed'
    if name.startswith('conv_proj'):
        name = name.replace('conv_proj', 'patch_embed.proj')
    if name.startswith('encoder.layers'):
        name = name.replace('encoder.layers.encoder_layer_', 'blocks.')
        name = name.replace('ln_', 'norm')
        name = name.replace('self_attention', 'attn')
        name = name.replace('in_proj_', 'qkv.')
        name = name.replace('out_proj', 'proj')
        name = name.replace('linear_', 'fc')
    if name.startswith('encoder.ln'):
        name = name.replace('encoder.ln', 'norm')
    if name.startswith('heads'):
        name = name.replace('heads.', '')
    timm_state_dict[name] = param
    
for k, v in timm_state_dict.items():
    print(k, v.shape)