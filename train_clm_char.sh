#nvidia-docker run -it --rm -v `pwd`:/stage/ -u $(id -u):$(id -g) bmccann/cuda10_th1_tf113:latest bash -c "cd stage && 

export NGPU=2
OUTPATH=data/text8

BATCH=64
CLIP=1
CTX=256
EMB=768
HID=3072
HEAD=12
L=12
EXPID=T,gelu,batch=${BATCH},clip=${CLIP},c=${CTX},emb=${EMB},hid=${HID},h=${HEAD},l=${L} 

#python -m torch.distributed.launch --nproc_per_node=$NGPU 
CUDA_VISIBLE_DEVICES=$1 python train.py --exp_name xlm_en --exp_id $EXPID --dump_path ./dumped  --data_path $OUTPATH --lgs 'en' --clm_steps 'en' --mlm_steps '' --emb_dim $EMB --n_layers $L --n_heads $HEAD --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $BATCH --bptt $CTX --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00020,weight_decay=0 --epoch_size 100000 --max_epoch 100000 --validation_metrics _valid_en_clm_ppl --stopping_criterion _valid_en_clm_ppl,25 --fp16 true --amp 2 --word_mask_keep_rand '0.0,1.0,0.0' --word_pred '0.99' --data_type character --clip_grad_norm $CLIP


## There are other parameters that are not 
