# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch \
        --log_dir=./debug/ --gpus '0,1'  \
        tools/train.py \
        -c configs/PP-OCRv5_server_rec_tyjt.yml
