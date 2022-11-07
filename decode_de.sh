# x=test
# fairseq-generate /home/sxy/Projects/cp/wmt_stand_19 \
# --path /home/sxy/Projects/cp/checkpoints/itnet/checkpoint33.pt \
# --task image_translation \
# --arch itransformer \
# --batch-size 8 \
# --dataset-impl raw \
# --remove-bpe \
# --beam 5 >/home/sxy/Projects/cp/checkpoints/itnet/temp.txt \


# x=test
# fairseq-generate /home/sxy/Projects/cp/base_data/en-zh/binary \
# --path /home/sxy/Projects/cp/checkpoints/text_model100000/checkpoint_6_90000.pt \
# --task translation \
# --arch transformer \
# --batch-size 8 \
# --gen-subset  "test" \
# --bpe  "sentencepiece"  --sentencepiece-model  "/home/sxy/Projects/cp/base_data/en-zh/english.model" \
# --scoring "sacrebleu" \
# --skip-invalid-size-inputs-valid-test \
# --beam 5 > /home/sxy/Projects/cp/base_data/en-zh/binary/temp.txt \




# x=test
fairseq-generate /home/sxy/Projects/cp/base_data/en-zh/contrastive \
--path /home/sxy/Projects/cp/checkpoints/duibi_model/checkpoint_best.pt \
--task image_translation \
--img-path /home/sxy/Projects/cp/base_data/img_data/ \
--arch contrastive \
--batch-size 8 \
--gen-subset  "test1" \
--bpe  "sentencepiece"  --sentencepiece-model  "/home/sxy/Projects/cp/base_data/en-zh/english.model" \
--scoring "sacrebleu" \
--skip-invalid-size-inputs-valid-test \
--beam 5 > /home/sxy/Projects/cp/base_data/en-zh/contrastive/temp3.txt \


# fairseq-generate /home/sxy/Projects/cp/base_data/en-zh/contrastive \
# --path /home/sxy/Projects/cp/checkpoints/itnet/checkpoint_6_80000.pt \
# --task image_translation \
# --img-path /home/sxy/Projects/cp/base_data/img_data/ \
# --arch itransformer \
# --batch-size 8 \
# --gen-subset  "test" \
# --bpe  "sentencepiece"  --sentencepiece-model  "/home/sxy/Projects/cp/base_data/en-zh/english.model" \
# --scoring "sacrebleu" \
# --skip-invalid-size-inputs-valid-test \
# --beam 5 > /home/sxy/Projects/cp/checkpoints/itnet/temp.txt \