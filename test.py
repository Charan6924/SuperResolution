import torch
from Generator import Generator


generator = Generator().to('cpu')
checkpoint = torch.load('checkpoints/pretrain_final.pt', map_location='cpu')
ckpt_keys = list(checkpoint['generator'].keys())[:3]
model_keys = list(generator.state_dict().keys())[:3]
print(f"Checkpoint keys: {ckpt_keys}")
print(f"Model keys: {model_keys}")