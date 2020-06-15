CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=3000 --dataset='reddit' --loss_norm=1 --eval_sample=1 --train_sample=1

#CUDA_VISIBLE_DEVICES=3 python main.py --batch_size=6000 --loss_norm=1 --eval_sample=0