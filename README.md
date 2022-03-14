# Tabbie: Tabular Information Embedding
This repository is forked from https://github.com/SFIG611/tabbie.git, which includes scripts for Tabbie(Tabular Information Embedding) model. 
The link to the paper is as follows.
https://arxiv.org/pdf/2105.02584.pdf

This repository is only for the use of our team project in the fall semester 2021 at the University of Mannheim with the topic of Data Integration using Deep Learning. 


## (setup 1): update cuda version from 10.0 to 10.1 (for AWS deep learning ami)
```
# https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
```

## setup 2: create conda env (default env name: "table_emb_dev")
```
git clone https://github.com/SFIG611/tabbie.git
cd tabbie
conda env create --file env/env.yml
conda activate table_emb_dev
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## setup 3: installing apex (gcc==5.4, cuda==10.1.168, cudnn==7.6-cuda_10.1)
```
conda activate table_emb_dev
mkdir -p third_party
git clone -q https://github.com/NVIDIA/apex.git third_party/apex
cd third_party/apex
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./
```

## setup 4: download model
```
https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0
mv freq.tar.gz mix.tar.gz tabbie/model
```


# finetuning with column type prediction
```
conda activate table_emb_dev
cd tabbie
python train.py --train_csv_dir ./data/ft_sato/train50.jsonl --train_label_path ./data/ft_sato/label.csv
python pred.py --test_csv_dir ./data/ft_sato/test10.jsonl --model_path ./out_model/model.tar.gz
python evaluate.py --test_csv_dir ./data/ft_sato/test10.jsonl --model_path ./out_model/model.tar.gz
```

# finetuning tables with cell labels
```
conda activate table_emb_dev
cd tabbie
python train.py --train_csv_dir ./data/ft_cell/train_csv --train_label_path ./data/ft_cell/train_label.csv
python pred.py --test_csv_dir ./data/ft_cell/test_csv --model_path ./out_model/model.tar.gz
```

# finetuning tables with column labels
```
conda activate table_emb_dev
cd tabbie
python train.py --train_csv_dir ./data/ft_col/train_csv --train_label_path ./data/ft_col/train_label.csv
python pred.py --test_csv_dir ./data/ft_col/test_csv --model_path ./out_model/model.tar.gz
```

# finetuning tables with table labels
```
conda activate table_emb_dev
cd tabbie
python train.py --train_csv_dir ./data/ft_table/train_csv --train_label_path ./data/ft_table/train_label.csv
python pred.py --test_csv_dir ./data/ft_table/test_csv --model_path ./out_model/model.tar.gz
```











