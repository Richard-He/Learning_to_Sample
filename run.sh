CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'ppi-large' --sampler rw --meta_sampler_type normalized 

#CUDA_VISIBLE_DEVICES=3 python main.py --batch_size=6000 --loss_norm=1 --eval_sample=0