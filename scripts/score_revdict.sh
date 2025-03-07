export CODWOE_DIR="/home/codwoe"

python3 $CODWOE_DIR/baseline_archs/code/score.py \
  $CODWOE_DIR/baseline_archs/models/revdict-ngram/$1/revdict_predictions_$1.json \
  --reference_files_dir $CODWOE_DIR/data/reference_data \
  --output_file $CODWOE_DIR/baseline_archs/models/revdict-ngram/$1/scores.txt
