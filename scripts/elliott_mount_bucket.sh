gcloud config set project dfd-mapping-poverty
[ -d /mnt/mounted_bucket ] || mkdir /mnt/mounted_bucket
sudo chmod a+w /mnt
sudo chmod a+w /mnt/mounted_bucket
gcsfuse dfd-poverty-bucket /mnt/mounted_bucket

