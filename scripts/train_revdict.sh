export CODWOE_DIR="/home/codwoe"

python3 $CODWOE_DIR/baseline_archs/code/revdict.py --do_train \
  --train_file $CODWOE_DIR/data/train-data_all/en.train.json \
  --dev_file $CODWOE_DIR/data/train-data_all/en.dev.json \
  --device cuda \
  --target_arch $1 \
  --summary_logdir $CODWOE_DIR/baseline_archs/logs/revdict-$2-ngram \
  --save_dir $CODWOE_DIR/baseline_archs/models/revdict-$2-ngram \
  --tokenizer $2
