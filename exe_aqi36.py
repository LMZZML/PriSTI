import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np

from dataset_aqi36 import get_dataloader
from main_model import GSTI_aqi36
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
    config["diffusion"]["adj_file"] = 'AQI36'
    config["seed"] = SEED

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/pm25_outsample_" + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"], device=args.device, val_len=args.val_len,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers,
        target_strategy=args.targetstrategy,
    )
    model = GSTI_aqi36(config, args.device).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))

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
    parser = argparse.ArgumentParser(description="GSTI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=4, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
    )
    parser.add_argument(
        "--val_len", type=float, default=0.1, help="index of month used for validation (value:[0-7])"
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")

    args = parser.parse_args()
    print(args)

    main(args)