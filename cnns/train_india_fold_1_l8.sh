python3 train.py \
  --verbose \
  --epochs=15 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_1_l8 \
  --data-subdir=fold_1 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --lr=1e-4 \
  --weight-decay=1e-5

