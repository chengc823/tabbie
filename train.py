import os
import json
import sys
import torch
import argparse
import tempfile
from pathlib import Path
from allennlp.commands import main
from scripts.bert import BertUtil

sys.path += ['./scripts', './table_embedder/dataset_readers', './table_embedder/models']
from scripts.util import Util

def cmd_builder(setting, template_path, overrides):
    setting.pop('cuda_devices')
    for name, val in setting.items():
        print(name, val)
        os.environ[name] = val
    os.environ["ALLENNLP_DEBUG"] = "TRUE"

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        template_path,
        "-s", setting['out_dir'],
        "--include-package", "table_embedder",
        "-o", overrides,
    ]


# def dump_jsonl(train_csvdir, valid_csvdir, train_label_path, valid_label_path, out_jsonl_dir, label_type):
#    train_path, valid_path = Path(out_jsonl_dir)/'train.jsonl', Path(out_jsonl_dir)/'valid.jsonl'
#    Util.csvdir_to_jsonl(Path(train_csvdir), train_path, label_path=train_label_path, label_type=label_type)
#    if valid_csvdir is not None:
#        Util.csvdir_to_jsonl(Path(valid_csvdir), valid_path, label_path=valid_label_path, label_type=label_type)
#    else:
#        valid_path = None
#    return train_path, valid_path


# rewrite to fit for jsonl input data
def dump_jsonl(train_jsondir, valid_jsondir, out_jsonl_dir):     # changes maded here
    train_path, valid_path = Path(out_jsonl_dir)/'train.jsonl', Path(out_jsonl_dir)/'valid.jsonl'
    with open(train_path, "w") as fout:
        with open(train_jsondir,'r') as json_file_train:
            new_lines_train = list(json_file_train)
            for line in new_lines_train:
                tmp = json.loads(line)
                fout.write(json.dumps(tmp)+'\n')
            
    if valid_jsondir is not None:
        with open(valid_path, "w") as fout:
            with open(valid_jsondir,'r') as json_file_valid:
                new_lines_valid = list(json_file_valid)
                for line in new_lines_valid:
                    tmp = json.loads(line)          
    else:
        valid_path = None
    return train_path, valid_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_cell/train_csv")
    parser.add_argument("--train_json_dir", help="input json dir for training", default="./data/ft_sato/train50.jsonl")
    # parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_table/train_csv")
    parser.add_argument("--train_label_path", help="train label path", default="./data/ft_sato/label.csv")
    # parser.add_argument("--train_label_path", help="train label path", default="./data/ft_col/train_label.csv")
    # parser.add_argument("--train_label_path", help="train label path", default="./data/ft_table/train_label.csv")
    parser.add_argument("--out_model_dir", help="output model dir", default="./out_model")
    parser.add_argument('--cache_cells', help="cache all initial cell emb", default=True, action='store_true')
    parser.add_argument("--valid_json_dir", help="input csv dir for validation")
    parser.add_argument("--valid_label_path", help="valid label path")
    parser.add_argument("--batch_size", help="batch size for finetuning", default="1")
    parser.add_argument("--cuda_devices", help="cuda device id list", default=[0])
    parser.add_argument("--config", help="config file for tabbie", default="./exp/ft_col/col.yml")
    args = parser.parse_args()

    # get label type
    # label_type = Util.get_label_type(args.train_label_path)

    # dump jsonl
    tmpdir = tempfile.TemporaryDirectory()
    # tmpdir = "./data/ft_sato"
    # label_type = "sato"
    # train_path, valid_path = dump_jsonl(args.train_json_dir, args.valid_json_dir, args.train_label_path, args.valid_label_path, tmpdir.name, label_type)      \
    
    train_path, valid_path = dump_jsonl(args.train_json_dir, args.valid_json_dir, tmpdir.name)      # changes maded here

    # setup params
    params = Util.load_yaml(args.config)
    params['bs'] = args.batch_size
    params['train_data_path'] = str(train_path)
    params['out_dir'] = args.out_model_dir
    params['cuda_devices'] = args.cuda_devices
    if os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        params['cuda_devices'] = [int(elem) for elem in os.getenv('CUDA_VISIBLE_DEVICES').split(',')]

    # dump cell cache
    if args.cache_cells:
        bert_util = BertUtil(True)
        bert_util.dump_emb(train_path, Path(tmpdir.name))
    #    os.environ['cache_dir'] = tmpdir.name
        del bert_util
        torch.cuda.empty_cache()

    # bert_util = BertUtil(True)
    # bert_util.dump_emb(train_path, Path(tmpdir.name))
    # #    os.environ['cache_dir'] = tmpdir.name
    # del bert_util
    # torch.cuda.empty_cache()

    os.environ['cache_dir'] = tmpdir.name
    # run allennlp command
    if args.valid_json_dir is not None:
        overrides = json.dumps({"distributed": {"cuda_devices": params['cuda_devices']}, "validation_data_path": str(valid_path)}) if len(params['cuda_devices']) > 1 else {}
    else:
        overrides = json.dumps({"distributed": {"cuda_devices": params['cuda_devices']}}) if len(params['cuda_devices']) > 1 else {}

    # cmd_builder(params, "exp/{}/{}.jsonnet".format(label_type, label_type), overrides)
    cmd_builder(params, "exp/sato/sato.jsonnet", overrides)
    main()



