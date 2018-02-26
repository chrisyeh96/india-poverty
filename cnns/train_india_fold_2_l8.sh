python3 train.py \
  --verbose \
  --epochs=15 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_2_l8_pov \
  --data-subdir=fold_2 \
  --label=secc_pov_rate \
  --fine-tune \
  --lr=1e-4 \
  --weight-decay=1e-4

