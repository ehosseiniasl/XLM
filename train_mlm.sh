#nvidia-docker run -it --rm -v `pwd`:/stage/ -u $(id -u):$(id -g) bmccann/cuda10_th1_tf113:latest bash -c "cd stage && 
CUDA_VISIBLE_DEVICeS=$1
OUTPATH=data/wiki/processed/XLM_en/30k 
python train.py --exp_name xlm_en --exp_id 0 --dump_path ./dumped  --data_path $OUTPATH --lgs 'en' --clm_steps '' --mlm_steps 'en' --emb_dim 2048 --n_layers 12 --n_heads 16 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 8 --bptt 256 --optimizer adam,lr=0.0001 --epoch_size 300000 --max_epoch 100000 --validation_metrics _valid_en_mlm_ppl --stopping_criterion _valid_en_mlm_ppl,25 --fp16 true --amp 2 --word_mask_keep_rand '0.8,0.1,0.1' --word_pred '0.15' --data_type word 


## There are other parameters that are not 
