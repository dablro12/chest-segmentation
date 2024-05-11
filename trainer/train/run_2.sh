python ./run.py --model "r2unet" \
                --version "v1" \
                --save_path "/mnt/HDD/chest-seg_models" \
                --cuda "0"\
                --ts_batch_size 64\
                --vs_batch_size 8\
                --epochs 1000\
                --loss "BCE"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --pretrain "no" --pretrained_model "premodel" --error_signal no\
                --wandb "yes"\ > r2unet.log 2>&1 &


