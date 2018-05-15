python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_1_l8_frac_32 \
  --data-subdir=fold_1 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_1_s1_frac_32 \
  --data-subdir=fold_1 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_2_l8_frac_32 \
  --data-subdir=fold_2 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=15 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_2_s1_frac_32 \
  --data-subdir=fold_2 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-5

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_3_l8_frac_32 \
  --data-subdir=fold_3 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_3_s1_frac_32 \
  --data-subdir=fold_3 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_4_l8_frac_32 \
  --data-subdir=fold_4 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_4_s1_frac_32 \
  --data-subdir=fold_4 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=l8 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_5_l8_frac_32 \
  --data-subdir=fold_5 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

python3 train.py \
  --verbose \
  --epochs=5 \
  --sat-type=s1 \
  --country=india \
  --log-epoch-interval=1 \
  --name=fold_5_s1_frac_32 \
  --data-subdir=fold_5 \
  --label=secc_cons_per_cap_scaled \
  --fine-tune \
  --train-frac=0.03125 \
  --lr=1e-5 \
  --weight-decay=1e-3

