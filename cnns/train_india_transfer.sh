python3 train.py \
  --verbose \
  --epochs=15 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=india_s1_2015_transfer \
  --label=secc_cons_per_cap_scaled \
  --no-fine-tune \
  --lr=1e-4 \
  --preload-model=india_s1_2015_dmsp

