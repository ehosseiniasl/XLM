#nvidia-docker run -it --rm -v `pwd`:/stage/ -u $(id -u):$(id -g) bmccann/cuda10_th1_tf113:latest bash -c "cd stage && 

NGPU=2
OUTPATH=data/text8 
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --exp_name xlm_en --dump_path ./dumped  --data_path $OUTPATH --lgs 'en' --clm_steps 'en' --mlm_steps '' --emb_dim 2048 --n_layers 12 --n_heads 16 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 8 --bptt 512 --optimizer adam,lr=0.0001 --epoch_size 300000 --max_epoch 100000 --validation_metrics _valid_en_clm_ppl --stopping_criterion _valid_en_clm_ppl,25 --fp16 true --amp 1 --word_mask_keep_rand '0.0,1.0,0.0' --word_pred '0.99' --data_type character


## There are other parameters that are not 
