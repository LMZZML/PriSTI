import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np

from dataset_pemsbay import get_dataloader
from main_model import PriSTI_PemsBAY
from utils import train, evaluate


def main(args):
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["diffusion"]["adj_file"] = 'pems-bay'
    config["seed"] = SEED

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/pemsbay_" + args.missing_pattern + '_' + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 载入数据
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers, target_strategy=args.targetstrategy
    )
    model = PriSTI_PemsBAY(config, args.device).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="traffic.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=4, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--missing_pattern", type=str, default="block")     # block|point

    args = parser.parse_args()
    print(args)

    main(args)