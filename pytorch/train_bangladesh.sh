python3 train.py \
  --verbose \
  --epochs=50 \
  --sat-type=s1 \
  --country=bangladesh \
  --log_epoch_interval=5 \
  --name=bangladesh_g_s1_2015_pretrained \
  --lr=1e-4 \
  --preload-model=india_s1_2015_scaled_latest \
  --use-grouped-labels
