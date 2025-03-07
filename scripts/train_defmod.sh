export CODWOE_DIR="/home/codwoe"

python3 $CODWOE_DIR/baseline_archs/code/defmod.py --do_train \
  --train_file $CODWOE_DIR/data/train-data_all/en.train.json \
  --dev_file $CODWOE_DIR/data/train-data_all/en.dev.json \
  --device cuda \
  --source_arch char \
  --summary_logdir $CODWOE_DIR/baseline_archs/logs/defmod-baseline \
  --save_dir $CODWOE_DIR/baseline_archs/models/defmod-baseline \
  --spm_model_path $CODWOE_DIR/baseline_archs/models/defmod-baseline
