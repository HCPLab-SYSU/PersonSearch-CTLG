# train
python3 train.py --model ViT_Bert_Overlap --gpus 0,1,2,3 --batch_size 256 --name exp_base --temperature 0.005 --cap_aug --stride_size 12 --mixup --dataset CUHKPEDES  --anno_path CUHK-PEDES/new_CUHK_json_1.json



