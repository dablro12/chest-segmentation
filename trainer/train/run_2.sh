python ./run.py --model "unet_plus_plus" \
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
                --wandb "yes"\ > output2.log 2>&1 &


