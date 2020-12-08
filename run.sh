bash

export DATA_DIR=/media/ganjinzero/Code/bert/data/
export BERT_BASE_DIR=/media/ganjinzero/Code/bert/chinese_L-12_H-768_A-12

python run_classifier.py \
   --task_name=mytask \
   --do_train=true \
   --do_eval=true \
   --data_dir=$DATA_DIR/ \
   --vocab_file=$BERT_BASE_DIR/vocab.txt \
   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
   --max_seq_length=128 \
   --train_batch_size=32 \
   --learning_rate=2e-5 \
   --num_train_epochs=3.0 \
   --output_dir=./data/output_models/