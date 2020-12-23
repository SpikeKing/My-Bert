export DATA_DIR=./data/MRPC
export BERT_BASE_DIR=./data/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=./data/output_models_mrpc/

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=./tmp/mrpc_output/