# ============================================================================
# == Main script to run experiments.
# 
# Note that this script will take about 5 days to run on a Tesla K80, due to
# the massive size (N ~= 550k) of the India dataset. We provide pre-trained
# models in the models/ directory which can be evaluated without needing to
# re-train any more models.
#
# This script will
# 1. stratify the datasets into train/valid/test
# 2. run baseline models and report R^2
# 3. train Sentinel-1 RGB CNN models for Bangladesh and India
# 4. report the R^2 for CNN models
# 
# Our plots were done in Jupyter notebook, see evaluation/eval.ipynb.
# Our heatmaps were written in R, see visualizations/R-script.
#
# 
# 12/11/2017

# == mount bucket

bash scripts/mount_staff_bucket.sh

# == baselines

cd baselines/
python3 stratify_datasets.py
python3 run_baselines.py

# == cnns

cd ../cnns
bash train_bangladesh.sh
bash train_india.sh

# == evaluation

cd ../evaluation
python3 eval.py --name india_s1_2015
python3 eval.py --name bangladesh_v_s1_2015 --use-grouped-labels


