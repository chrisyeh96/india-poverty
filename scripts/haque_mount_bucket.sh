gcloud config set project poverty-prediction
[ -d /mnt/mounted_bucket ] || mkdir /mnt/mounted_bucket
sudo chmod a+w /mnt
sudo chmod a+w /mnt/mounted_bucket
gcsfuse poverty-bucket /mnt/mounted_bucket
