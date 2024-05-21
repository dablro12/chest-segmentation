python ./run.py --model "monai_swinunet" \
                --version "v12" \
                --save_path "/mnt/HDD/chest-seg_models" \
                --cuda "0"\
                --ts_batch_size 96\
                --vs_batch_size 32\
                --epochs 1000\
                --loss "DiceCELoss"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler None\
                --pretrain "no" --pretrained_model "premodel" --error_signal no\
                --wandb "yes"\ > monai_swinunet.log 2>&1 &