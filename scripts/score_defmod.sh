export CODWOE_DIR="/home/codwoe"

python3 $CODWOE_DIR/baseline_archs/code/score.py \
  $CODWOE_DIR/baseline_archs/models/defmod-baseline/char/defmod_predictions_char.json \
  --reference_files_dir $CODWOE_DIR/data/reference_data \
  --output_file $CODWOE_DIR/baseline_archs/models/defmod-baseline/char/scores.txt
