from dataclasses import dataclass


@dataclass
class TrainingConfig:
    pretrain_epochs: int = 50
    gan_epochs: int = 100
    batch_size: int = 256
    num_workers: int = 3

    lr_g: float = 1e-4
    lr_d: float = 1e-6

    weight_adversarial: float = 0.005
    weight_pixel: float = 1.0
    weight_ssim: float = 0.3

    pretrain_ssim_weight: float = 0.1

    grad_clip_norm: float = 10.0
    discriminator_update_freq: int = 3

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    tensor_dir: str = 'data/pt_tensors'
    checkpoint_dir: str = 'checkpoints'
    samples_dir: str = 'samples'

    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
