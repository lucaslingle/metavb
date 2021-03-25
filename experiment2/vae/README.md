

CIFAR-10:
```
python app.py \
    --mode=train \
    --dataset=cifar10 \
    --img_height=32 \
    --img_width=32 \
    --img_channels=3 \
    --discrete_outputs=False \
    --batch_size=16 \
    --episode_len=64 \
    --code_size=200 \
    --num_filters=32 \
    --checkpoint_dir=checkpoints/vae_cifar10_b16t64c200f32_lr0pt001/ \
    --load_checkpoint=checkpoints/vae_cifar10_b16t64c200f32_lr0pt001/ \
    --summaries_dir=tensorboard_logs/vae_cifar10_b16t64c200f32_lr0pt001/ \
    --output_dir=output/ \
    --use_bn=True \
    --lr=0.001 \
    --epochs=100
```

CelebA:
```
python app.py \
    --mode=train \
    --dataset=celeb_a \
    --img_height=32 \
    --img_width=32 \
    --img_channels=3 \
    --discrete_outputs=False \
    --batch_size=16 \
    --episode_len=64 \
    --code_size=200 \
    --num_filters=32 \
    --checkpoint_dir=checkpoints/vae_celeba_b16t64c200f32_lr0pt001/ \
    --load_checkpoint=checkpoints/vae_celeba_b16t64c200f32_lr0pt001/ \
    --summaries_dir=tensorboard_logs/vae_celeba_b16t64c200f32_lr0pt001/ \
    --output_dir=output/ \
    --use_bn=True \
    --lr=0.001 \
    --epochs=60
```
