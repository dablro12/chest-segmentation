python ./run.py --model "unet" \
                --version "v7" \
                --save_path "/mnt/HDD/chest-seg_models" \
                --cuda "0"\
                --ts_batch_size 16\
                --vs_batch_size 4\
                --epochs 1000\
                --loss "BCEwithlogits"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler "ReduceLROnPlateau"\
                --pretrain "no" --pretrained_model "premodel" --error_signal no\
                --wandb "yes"\ > unet.log 2>&1 &


