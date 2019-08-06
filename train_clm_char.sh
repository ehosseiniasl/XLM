#nvidia-docker run -it --rm -v `pwd`:/stage/ -u $(id -u):$(id -g) bmccann/cuda10_th1_tf113:latest bash -c "cd stage && 

export NGPU=2
OUTPATH=data/text8 
#python -m torch.distributed.launch --nproc_per_node=$NGPU 
python train.py --exp_name xlm_en --exp_id context_512 --dump_path ./dumped  --data_path $OUTPATH --lgs 'en' --clm_steps 'en' --mlm_steps '' --emb_dim 768 --n_layers 12 --n_heads 12 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 4 --bptt 512 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00020,weight_decay=0 --epoch_size 100000 --max_epoch 100000 --validation_metrics _valid_en_clm_ppl --stopping_criterion _valid_en_clm_ppl,25 --fp16 true --amp 2 --word_mask_keep_rand '0.0,1.0,0.0' --word_pred '0.99' --data_type character --clip_grad_norm 1


## There are other parameters that are not 
