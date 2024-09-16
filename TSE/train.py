import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import sys
import yaml


from base.solver import Solver
from models.bsrnn.bsrnn import BSRNN
from models.dpccn.dpccn import DPCCN
from losses import singlesrc_neg_sisdr

model_dict = {
    "bsrnn": BSRNN,
    "dpccn": DPCCN,
}

loss_dict = {
    "sisdr": singlesrc_neg_sisdr,
}


class CriterionSystem(nn.Module):
    def __init__(self, criterion):
        super(CriterionSystem, self).__init__()
        self.criterion = loss_dict[criterion]

    def forward(self, estimates, targets, training=False):
        loss = self.criterion(estimates, targets)
        if not training:
            return (loss,)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgpath", type=str, required=True)
    args = parser.parse_args()
    with open(args.cfgpath) as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    # if cfg['distribute']['distributed']:
    #     if cfg['distribute'].get('gpu_ids', None) is None:
    #         logging.error("set your gpu codes like [0,1]")
    #     torch.cuda.set_device(local_rank)
    #     dist.init_process_group(backend='nccl')
    solver = Solver(
        valid_num=1,
        save_home=cfg["training"]["exp_dir"],
        load_param_index=cfg["training"]["load_param_index"],
        eval_interval=cfg["training"]["eval_interval"],
        num_avg=cfg["training"]["num_avg"],
        accum_grad=cfg["training"]["accum_grad"],
        clip_grad=cfg["grad_clipping"]["clip_grad"],
        **cfg["data"],
        **cfg["distribute"],
        **cfg["optim"],
    )

    # # model
    model = model_dict[cfg["training"]["model_name"]](
        **cfg["model"],
    )
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Num of Trainable Params = {total_params}")

    # # loss
    criterion = CriterionSystem(cfg["training"]["loss"])

    # # init
    print("Init solver")
    if cfg["distribute"]["distributed"]:
        solver.init(
            # ModelSystem
            model=model,
            criterion=criterion,
            local_rank=local_rank,
        )
    else:
        solver.init(
            # ModelSystem
            model=model,
            criterion=criterion,
            gpus=cfg["distribute"]["gpu_ids"],
        )

    with open(f"{cfg['training']['exp_dir']}/cfg.yaml", "w") as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
    start = time.time()
    print("Start training")
    solver.train()
    print("cost:", (time.time() - start) / 3600)
