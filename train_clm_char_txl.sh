#nvidia-docker run -it --rm -v `pwd`:/stage/ -u $(id -u):$(id -g) bmccann/cuda10_th1_tf113:latest bash -c "cd stage && 

OUTPATH=data/text8

BATCH=22
CLIP=0.25
CTX=256
EMB=512
HID=2048
HEAD=8
L=12
MEM=256
MODEL=transformer_xl
ATTN=0
ATTNDROP=0.0
DROP=0.1

EXPID=model=${MODEL},relu,batch=${BATCH},clip=${CLIP},c=${CTX},emb=${EMB},hid=${HID},h=${HEAD},l=${L},mem=${MEM},attn=${ATTN},attn_drop=${ATTNDROP} 

#export NGPU=2
#python -m torch.distributed.launch --nproc_per_node=$NGPU 
CUDA_VISIBLE_DEVICES=$1 python train.py --exp_name xlm_en --exp_id $EXPID --dump_path ./dumped  --data_path $OUTPATH --lgs 'en' --clm_steps 'en' --mlm_steps '' --emb_dim $EMB --n_layers $L --n_heads $HEAD --dropout $DROP --attention_dropout $ATTNDROP --gelu_activation false --batch_size $BATCH --bptt $CTX --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00020,weight_decay=0 --epoch_size 100000 --max_epoch 100000 --validation_metrics _valid_en_clm_ppl --stopping_criterion _valid_en_clm_ppl,25 --fp16 true --amp 2 --word_mask_keep_rand '0.0,1.0,0.0' --word_pred '0.99' --data_type character --clip_grad_norm $CLIP --model_type $MODEL --mem_len $MEM --attn_type $ATTN

## There are other parameters that are not 
