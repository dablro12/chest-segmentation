python ./run.py --model "monai_swinunet" \
<<<<<<< HEAD
                --version "v7" \
                --save_path "/mnt/HDD/chest-seg_models" \
                --cuda "0"\
                --ts_batch_size 128\
                --vs_batch_size 64\
                --epochs 2000\
                --loss "DiceCELoss"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler None\
                --pretrain "yes" --pretrained_model "/mnt/HDD/chest-seg_models/monai_swinunet_v7_240519/model_1100.pt" --error_signal no\
                --wandb "no"\ > monai_swinunet.log 2>&1 &
=======
                --version "v2" \
                --save_path "/mnt/HDD/oci-seg_models" \
                --cuda "0"\
                --ts_batch_size 24\
                --vs_batch_size 8\
                --epochs 400\
                --loss "BCE+DiceCELoss"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler None\
                --pretrain "no" --pretrained_model ".pt" --error_signal no\
                --wandb "yes"\ > monai_swinunet.log 2>&1 &
>>>>>>> 4f34f26cde23ccf0785f5fc8b7ae5da11ebe5111
