export CODWOE_DIR="/home/codwoe"

python3 $CODWOE_DIR/baseline_archs/code/revdict.py --do_pred \
  --test_file $CODWOE_DIR/data/test-data_all/en.test.revdict.json \
  --device cuda \
  --target_arch $1 \
  --save_dir $CODWOE_DIR/baseline_archs/models/revdict-ngram/$1 \
  --pred_file $CODWOE_DIR/baseline_archs/models/revdict-ngram/$1/revdict_predictions_$1.json
