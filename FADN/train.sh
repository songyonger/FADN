nohup python -u main.py --model FADN --scale 2 --n_feats 64 --rgb_range 255 --patch_size 96 --n_resblocks 3 --save FADN_X2 --data_train DIV2K --data_test DIV2K --dir_data /home/ --ext bin --gpu_id 6 --chop --epochs 800 --lr_decay 250 --n_GPUs 1 --loss 1*L1 --lr 1e-4 > train_x2log.txt