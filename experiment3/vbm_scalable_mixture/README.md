
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
    --num_clusters=10 \
    --memory_size=6 \
    --code_size=200 \
    --num_filters=64 \
    --opt_iters=10 \
    --checkpoint_dir=checkpoints/vbmgmm_cifar10_b16t64h10k6c200l10f64_lr0pt0005_beta1eqZERO_gceq100_kmpp_swishgroupencdec_nsr/ \
    --load_checkpoint=checkpoints/vbmgmm_cifar10_b16t64h10k6c200l10f64_lr0pt0005_beta1eqZERO_gceq100_kmpp_swishgroupencdec_nsr/ \
    --summaries_dir=tensorboard_logs/vbmgmm_cifar10_b16t64h10k6c200l10f64_lr0pt0005_beta1eqZERO_gceq100_kmpp_swishgroupencdec_nsr/ \
    --output_dir=output/ \
    --use_bn=False \
    --trainable_memory=False \
    --sample_memory=True \
    --sr_alpha=8.0 \
    --sr_beta=8.0 \
    --sr_gamma=0.50 \
    --sr_delta=0.20 \
    --sr_epsilon=0.10 \
    --lr=0.0005 \
    --epochs=200
```
Then reload checkpoint by invoking a similar command as above, and set ```--epochs=10 --lr=0.0001 --sr_alpha=8.0 --sr_beta=8.0 --sr_gamma=0.50 --sr_delta=0.000002 --sr_epsilon=0.000001```.

After that, you can resize the model by changing the ```--num_clusters``` flag and you can change the episode length by changing the ```--episode_len``` flag. 
To generate from memory, use ```--mode=generate```. 

You can see the full list of supported modes by running ```app.py --help```, and a printout will appear. 
