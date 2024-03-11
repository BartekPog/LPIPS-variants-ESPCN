import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../data/DIV2K/original --output_dir ../data/DIV2K/ESPCN/train --image_size 70 --step 35 --num_workers 16")

# Split train and valid
os.system("python3 ./split_train_valid_dataset.py --train_images_dir ../data/DIV2K/ESPCN/train --valid_images_dir ../data/DIV2K/ESPCN/valid --valid_samples_ratio 0.1")



# # Prepare dataset
# os.system("python3 ./prepare_dataset.py --images_dir ../data/T91/original --output_dir ../data/T91/ESPCN/train --image_size 70 --step 35 --num_workers 16")

# # Split train and valid
# os.system("python3 ./split_train_valid_dataset.py --train_images_dir ../data/T91/ESPCN/train --valid_images_dir ../data/T91/ESPCN/valid --valid_samples_ratio 0.1")
