python3 train.py \
  --verbose \
  --epochs=15 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_2_s1 \
  --data-subdir=fold_2 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --lr=1e-5 \
  --weight-decay=1e-5

