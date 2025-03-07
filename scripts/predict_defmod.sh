export CODWOE_DIR="/home/codwoe"

python3 $CODWOE_DIR/baseline_archs/code/defmod.py --do_pred \
  --test_file $CODWOE_DIR/data/test-data_all/en.test.defmod.json \
  --device cuda \
  --source_arch char \
  --save_dir $CODWOE_DIR/baseline_archs/models/defmod-baseline/char \
  --pred_file $CODWOE_DIR/baseline_archs/models/defmod-baseline/char/defmod_predictions_char.json
