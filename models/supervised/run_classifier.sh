#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  python3 src/classifier/multilabel_classifier.py \
      --do_train \
      --model_name_or_path bert-base-uncased \
      --train_file data/epfl_refined/train.json \
      --validation_file data/epfl/dev_final.json \
      --text_column_name tokens \
      --label_column_name labels \
      --output_dir 1412-esco-multilabel-epfl-train-refined-epfl/$SEED/bert-base-uncased \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --learning_rate 3e-5 \
      --checkpointing_steps epoch \
      --num_train_epochs 100 \
      --patience 100 \
      --seed $SEED \
      --batch_loss \
      --max_length 128
      
done