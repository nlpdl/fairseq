# fairseq-train /home/sxy/Projects/cp/wmt_stand_19 \
#   --no-progress-bar \
#   --log-format simple \
#   --task image_translation \
#   --arch itransformer \
#   --save-dir /home/sxy/Projects/cp/checkpoints/itnet \
#   --dropout 0.1 \
#   --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps '1e-9' --clip-norm 0.0 \
#   --warmup-updates 8000 --lr-scheduler inverse_sqrt  --lr 0.0007 \
#   --max-tokens 4096 \
#   --dataset-impl raw \
#   --patience 50 \
#   --num-workers 8 \
#   --batch-size 16 \
#   --criterion label_smoothed_cross_entropy --label-smoothing 0.1  --weight-decay 0.0 \
#   --finetune-from-model /home/sxy/Projects/cp/checkpoints/baseline/checkpoint50.pt \
#   --encoder-embed-dim 256 \
#   --attention-dropout 0.1 \
#   --activation-dropout 0.1 \
#   --adaptive-softmax-dropout 0.1  &


dest=/home/sxy/Projects/cp/base_data/en-zh/contrastive
model_path="/home/sxy/Projects/cp/checkpoints/itnet"
fairseq-train $dest -s "zh" -t "en" \
--save-dir $model_path \
--save-interval-updates 10000 \
--no-epoch-checkpoints \
--arch itransformer \
--batch-size 16 \
--patience 10 \
--task image_translation \
--optimizer adam --adam-betas "(0.9, 0.98)" --lr 1e-4 --lr-scheduler inverse_sqrt \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 \
--max-tokens 1024 --update-freq "4" --skip-invalid-size-inputs-valid-test --log-interval 10 &

# --finetune-from-model /home/sxy/Projects/cp/checkpoints/baseline/checkpoint50.pt \