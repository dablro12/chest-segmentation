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
python ./run.py --model "manet" \
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
                --wandb "yes"\ > output.log 2>&1 &


