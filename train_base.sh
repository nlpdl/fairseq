# fairseq-train /home/sxy/newpan/CP/fairseq_e_t_e/fairseq/data_bin_new_zh_en \
#   --no-progress-bar \
#   --log-format simple \
#   --arch transformer \
#   --dropout 0.3 \
#   --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps '1e-9' --clip-norm 0.0 \
#   --warmup-updates 8000 --lr-scheduler inverse_sqrt  --lr 0.0007 \
#   --max-tokens 4096 \
#   --dataset-impl raw \
#   --eval-bleu \
#   --eval-bleu-args '{"beam":5, "max_len_a":1.2, "max_len_b":10}' \
#   --eval-bleu-detok moses \
#   --patience 10 \
#   --num-workers 8 \
#   --batch-size 64 \
#   --criterion label_smoothed_cross_entropy --label-smoothing 0.1  --weight-decay 0.0 \
#   --save-dir checkpoints/zh \
#   --best-checkpoint-metric bleu \
#   --maximize-best-checkpoint-metric \
#   --attention-dropout 0.1 \
#   --activation-dropout 0.1 \
#   --adaptive-softmax-dropout 0.1  &

# training 
dest="/home/sxy/Projects/cp/base_data/en-zh/binary"
model_path="/home/sxy/Projects/cp/checkpoints/text_model100000"

fairseq-train $dest \
    --save-dir $model_path \
    --source-lang "zh" \
    --target-lang "en" \
    --batch-size 64 \
    --save-interval-updates 10000 \
    --arch transformer \
    --task translation \
    --patience 10 \
    --no-epoch-checkpoints \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --dropout 0.1 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --log-interval 10 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --update-freq 8 --skip-invalid-size-inputs-valid-test \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 8 --decoder-attention-heads 8 --share-decoder-input-output-embed \
    --dataset-impl "mmap" \
    --ddp-backend "c10d" &
    # --max-update 10000 \
      # --no-epoch-checkpoints \