# TEXT=/home/sxy/newpan/CP/end_to_end/data/data_bin_de_wmt14_img
# fairseq-preprocess --source-lang en --target-lang de \
#   --nwordstgt 16000 --nwordssrc 16000\
#   --trainpref $TEXT/train.en-de \
#   --validpref $TEXT/valid.en-de \
#   --testpref $TEXT/test.en-de \
#   --destdir data_bin_de_wmt14_img/ \
#   --dataset-impl raw

# TEXT=/home/sxy/newpan/CP/zh-data/temp
# fairseq-preprocess --source-lang zh --target-lang en \
#   --nwordstgt 10000 --nwordssrc 10000\
#   --trainpref $TEXT/train \
#   --validpref $TEXT/valid \
#   --testpref $TEXT/test \
#   --destdir data_bin_clean_zh_en \
#   --dataset-impl raw

# TEXT=/home/sxy/newpan/CP/fairseq_e_t_e/fairseq/data_bin_zh_en
# fairseq-preprocess --source-lang cn --target-lang en \
#   --nwordstgt 16000 --nwordssrc 16000\
#   --trainpref $TEXT/train \
#   --validpref $TEXT/valid \
#   --testpref $TEXT/test \
#   --destdir data_bin_new_zh_en \
#   --dataset-impl raw

TEXT=/home/sxy/newpan/CP/data_bin_clean_zh_en
fairseq-preprocess --source-lang zh --target-lang en \
  --nwordstgt 16000 --nwordssrc 16000\
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir /home/sxy/newpan/CP/data_bin_my_zh_en \
  --dataset-impl raw