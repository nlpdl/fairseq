dest=/home/sxy/Projects/cp/base_data/en-zh/contrastive
model_path="/home/sxy/Projects/cp/checkpoints/duibi_model"
fairseq-train $dest -s "zh" -t "en" \
--save-dir $model_path \
--save-interval-updates 10000 \
--no-epoch-checkpoints \
--arch contrastive \
--batch-size 16 \
--patience 10 \
--task image_translation \
--optimizer adam --adam-betas "(0.9, 0.98)" --lr 1e-4 --lr-scheduler inverse_sqrt \
--criterion my_criterion --label-smoothing 0.1 --dropout 0.1 \
--max-tokens 1024 --update-freq "4" --skip-invalid-size-inputs-valid-test --log-interval 10 &

# --finetune-from-model "/home/sxy/Projects/cp/checkpoints/text_model100000/checkpoint_best.pt" \
# --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \

# fairseq-train data_bin_new_zh_en \
#   --no-progress-bar \
#   --log-format simple \
#   --task image_translation \
#   --arch contrastive \
#   --save-dir /home/sxy/Projects/cp/checkpoints/finetune \
#   --dropout 0.1 \
#   --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps '1e-9' --clip-norm 0.0 \
#   --warmup-updates 8000 --lr-scheduler inverse_sqrt  --lr 0.0007 \
#   --max-tokens 4096 \
#   --dataset-impl raw \
#   --patience 50 \
#   --num-workers 8 \
#   --batch-size 16 \
#   --criterion my_criterion --label-smoothing 0.1  --weight-decay 0.0 \
#   --encoder-embed-dim 256 \
#   --attention-dropout 0.1 \
#   --activation-dropout 0.1 \
#   --adaptive-softmax-dropout 0.1  &
    # --best-checkpoint-metric bleu \
  # --eval-bleu \
  # --eval-bleu-args '{"beam":5, "max_len_a":1.2, "max_len_b":10}' \
  # --eval-bleu-detok moses \
    # --finetune-from-model /home/sxy/newpan/CP/fairseq_e_t_e/fairseq/checkpoints/zh/checkpoint50.pt \
  # --maximize-best-checkpoint-metric \
  # --patience 10 \
  # --premodelpath '/home/sxy/newpan/CP/fairseq_e_t_e/fairseq/data_bin_zh_en/craft_mlt_25k.pth'
  # --save-dir checkpoints/base \
  # --best-checkpoint-metric bleu \
  # --eval-bleu \
  # --eval-bleu-args '{"beam":5, "max_len_a":1.2, "max_len_b":10}' \
  # --eval-bleu-detok moses \