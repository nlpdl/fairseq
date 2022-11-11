


image_to_text_base里的load_state_dict看情况切换
text-pretrain
```
DATA_ROOT=/home/sxy/Projects/cp/base_data/model_v2/my-model/data-bin
SAVE_DIR=/home/sxy/Projects/cp/workspace/checkpoints/text_pretrain

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --ddp-backend c10d \
  --no-progress-bar \
  --log-format simple \
  --find-unused-parameters \
  --finetune-from-model "/home/sxy/Projects/cp/checkpoints/text_character_model/checkpoint_10_170000.pt" \
  --seed 1337 \
  \
  --task image_pretrain_translation \
  --train-type text_pretrain \
  --label-rates 50 \
  --sample-rate 16000 \
  --random-crop \
  \
  --num-workers 0 \
  --max-tokens 1024 \
  \
  --criterion label_smoothed_cross_entropy \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-09 \
  --weight-decay 0.01 \
  --clip-norm 0.0 \
  --dropout 0.1 \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  \
  --batch-size 32 \
  --warmup-updates 8000 \
  --max-epoch 50 \
  \
  --arch imagetonetpretrain \
  --taskname text_pretrain \
  --input-channel 3 \
  --output-channel 256 \
```

img_pretrain
```
DATA_ROOT=/home/sxy/Projects/cp/base_data/model_v2/my-model/data-bin
SAVE_DIR=/home/sxy/Projects/cp/workspace/checkpoints/img_pretrain

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --ddp-backend c10d \
  --img-path /home/sxy/Projects/cp/base_data/model_v2/img_data/ \
  --no-progress-bar \
  --log-format simple \
  --find-unused-parameters \
  --continue-once "/home/sxy/Projects/cp/workspace/checkpoints/text_pretrain/checkpoint5.pt" \
  --seed 1337 \
  \
  --task image_pretrain_translation \
  --train-type img_pretrain \
  --label-rates 50 \
  --sample-rate 16000 \
  --random-crop \
  \
  --num-workers 0 \
  --max-tokens 1024 \
  \
  --criterion my_criterion \
  --optimizer adam \
  --reset-optimizer \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-09 \
  --weight-decay 0.01 \
  --clip-norm 0.0 \
  --dropout 0.1 \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  \
  --batch-size 16 \
  --warmup-updates 8000 \
  --max-epoch 10 \
  \
  --arch imagetonetpretrain \
  --taskname img_pretrain \
  --input-channel 3 \
  --output-channel 256 \
```