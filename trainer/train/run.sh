# python ./run_multi.py --model "AOTGAN-random-mask" \
#                 --version "v1" \
#                 --cuda "0"\
#                 --ts_batch_size 4\
#                 --vs_batch_size 2\
#                 --epochs 50\
#                 --loss "ce"\
#                 --optimizer "Adam"\
#                 --learning_rate 0.0001\
#                 --scheduler "lambda"\
#                 --pretrain "no" --pretrained_model "Places2" --error_signal no\
#                 --wandb "yes"\ > output.log 2>&1 &
python ./run.py --model "Swin-UNET" \
                --version "v1" \
                --save_path "/mnt/HDD/chest-seg_models/swin-unet" \
                --cuda "0"\
                --ts_batch_size 128\
                --vs_batch_size 128\
                --epochs 3000\
                --loss "logitBCE"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --pretrain "no" --pretrained_model "premodel" --error_signal no\
                --wandb "yes"\ > output.log 2>&1 &


