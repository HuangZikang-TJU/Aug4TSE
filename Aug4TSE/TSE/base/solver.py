import warnings
import sys

warnings.filterwarnings("ignore")
import os
import gc
import numpy as np
import logging
import torch
import torch.nn as nn
from torch import optim
import time
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
seed = 102
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.


class Solver(object):
    def __init__(
        self,
        valid_num=3,
        save_home="",
        dataset="librimix",
        load_param_index=-1,
        optimizer="",
        lr=1e-3,
        final_lr=None,
        lr_descend_factor=0.75,
        num_avg=5,
        stop_patience=10,
        patience=-1,
        weight_decay=0,
        eval_interval=5,
        accum_grad=1,
        clip_grad=None,
        max_epoch=999,
        **data_cfg,
    ):
        self.save_home = save_home
        self.device = torch.device("cuda:0")
        assert dataset == "librimix", f"Not support {dataset}"
        if dataset == "librimix":
            from data.librimix_loader import get_dataloader
        elif dataset == "wsj0mix":
            from data.wsj0mix_loader import get_dataloader
        else:
            raise NotImplementedError
        self.train_loader, self.val_loader = get_dataloader(
            **data_cfg,
        )

        self.load_param_index = load_param_index
        # ----------------------
        self.valid_losses_num = valid_num
        self.valid_losses = [[] for _ in range(self.valid_losses_num)]
        print(self.valid_losses)
        # ----------------------
        self.train_losses = []

        self.max_epoch = max_epoch
        self.lr = lr
        self.final_lr = final_lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_descend_factor = lr_descend_factor
        self.stop_patience = stop_patience

        self.saved_path = []
        self.criterion = None
        self.patience = patience
        self.lr_descend_flag = 0
        self.early_stop_flag = 0
        self.eval_interval = eval_interval
        self.clip_grad = clip_grad
        self.num_avg = num_avg
        self.local_rank = None
        self.accum_grad = accum_grad

    def init(
        self,
        model,
        criterion,
        optimizer=None,
        local_rank=None,
        gpus=None,
        tts_pipeline=None,
        speaker_pipeline=None,
    ):
        self.local_rank = local_rank
        self.criterion = criterion
        self.tts_pipeline = tts_pipeline
        self.speaker_pipeline = speaker_pipeline

        if gpus is None and local_rank is None:
            logging.error("set your gpu codes like [0,1] or set your local rank")
            return

        os.makedirs(self.save_home, exist_ok=True)
        # Reset model
        if self.load_param_index >= 0:  # TODO: to load previous checkpoints
            model_path = os.path.join(
                self.save_home, "_ckpt_epoch_%d.ckpt" % self.load_param_index
            )
            if not os.path.exists(model_path):
                logging.error("model param path error:" + model_path)
                return
            else:
                print(f"loading param {model_path}")
                ckpt = torch.load(model_path, map_location=self.device)
            try:
                model.load_state_dict(ckpt["model"], strict=True)
            except Exception as e:
                new_state_dict = {}
                for key, val in ckpt["model"].items():
                    if key.startswith("model"):
                        new_state_dict[key[6:]] = val
                model.load_state_dict(new_state_dict, strict=True)
            print("load param successful! " + str(self.load_param_index))

            self.valid_losses = ckpt["valid_losses"]
            self.saved_path = ckpt["saved_models"]

        if local_rank is not None:
            model.cuda()
            self.ddp_model = nn.parallel.DistributedDataParallel(model)
        # using nn.DataParallel
        else:
            print(gpus)
            self.ddp_model = nn.DataParallel(model, device_ids=gpus).cuda()
        # for name, param in self.ddp_model.named_parameters(): print(name, param)

        if optimizer is None:
            if self.optimizer == "adam":
                print("init adam")
                self.optimizer = optim.Adam(
                    list(self.ddp_model.module.parameters()),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                if self.load_param_index >= 0:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                    last = self.optimizer.state_dict()["param_groups"][0]["lr"]
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = last * self.lr_descend_factor
                    print(
                        "Learning rate adjusted from %f to %f; patience:%d"
                        % (
                            last,
                            self.optimizer.state_dict()["param_groups"][0]["lr"],
                            self.patience,
                        )
                    )
                    # print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            else:
                raise NotImplementedError
        else:
            self.optimizer = optimizer

    def train(self):
        print("train:%d,valid:%d" % (len(self.train_loader), len(self.val_loader)))
        start_epoch = 0
        if self.load_param_index > 0:
            start_epoch = self.load_param_index

        for epoch in range(start_epoch, self.max_epoch):
            # Training
            self.ddp_model.train()
            start = time.time()
            train_avg_loss = self._run_one_epoch(epoch, training=True)

            # Evaluating
            self.ddp_model.eval()
            # self.model.module.eval()
            valid_losses = None
            if epoch % self.eval_interval == 0 and epoch != 0:
                valid_losses = self._run_one_epoch(epoch, training=False)

            self.make_epoch_log(epoch, start, train_avg_loss, valid_losses)

            # Update valid loss
            if (
                valid_losses is not None
                and not self.valid_loss_update_and_check_better(valid_losses)
            ):
                print("Not better")
                # logging.info("Not better")
            else:
                self.save_model(epoch)

            if epoch == self.max_epoch - 1:
                self.make_final_log()
                print("Stop! Saving last checkpoint: ")
                self.save_model(epoch)
                break

            if (
                self.early_stop_flag >= self.stop_patience
                or self.optimizer.state_dict()["param_groups"][0]["lr"] <= self.final_lr
            ):  # reached stop_patience
                self.make_final_log()
                print("Stop! Saving last checkpoint: ")
                self.save_model(epoch)
                break

            if self.lr_descend_flag >= self.patience:
                last = self.optimizer.state_dict()["param_groups"][0]["lr"]
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = last * self.lr_descend_factor
                print(
                    "\nLearning rate adjusted from %f to %f; patience:%d"
                    % (
                        last,
                        self.optimizer.state_dict()["param_groups"][0]["lr"],
                        self.patience,
                    )
                )
                if epoch > 0:
                    self.make_final_log()
                self.lr_descend_flag = 0

    def _run_one_epoch(self, epoch, training=False):
        total_loss = 0
        valid_loss_total = [0 for _ in range(self.valid_losses_num)]
        data_loader = self.train_loader if training else self.val_loader
        batch_id = 0
        print(f"Epoch {epoch}, {'Training' if training else 'Validating'}")
        if training:
            self.optimizer.zero_grad()
        for batch_id, (wav, target_wavs, spk_embeddings, *others) in tqdm(
            enumerate(data_loader)
        ):
            if training:
                # get model outputs
                if self.local_rank is not None:
                    wav = wav.cuda()
                    spk_embeddings = spk_embeddings.cuda()
                model_outputs = self.ddp_model(wav, spk_embeddings)

                # compute losses
                batch_loss = self.criterion(
                    model_outputs, target_wavs.cuda(), training
                ).mean()
                if self.accum_grad > 1:
                    batch_loss = batch_loss / self.accum_grad
                if batch_loss.item() != batch_loss.item():  # check if nan?
                    continue

                # backpropogation
                batch_loss.backward()

                if self.clip_grad is not None:
                    nn.utils.clip_grad_norm(
                        self.ddp_model.parameters(),
                        max_norm=self.clip_grad,
                        norm_type=2,
                    )

                total_loss += batch_loss.item()
                if ((batch_id + 1) % self.accum_grad == 0) or (
                    (batch_id + 1) == len(data_loader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                with torch.no_grad():
                    # get model outputs
                    if self.local_rank is not None:
                        wav = wav.cuda()
                        spk_embeddings = spk_embeddings.cuda()
                    model_outputs = self.ddp_model(wav, spk_embeddings)

                    # compute losses
                    valid_losses = self.criterion(model_outputs, target_wavs.cuda())
                    for index in range(len(valid_loss_total)):
                        valid_loss_total[index] += valid_losses[index].mean().item()
        gc.collect()
        if training:
            return total_loss / (batch_id + 1)
        else:
            return np.array(valid_loss_total) / (batch_id + 1)

    def valid_loss_update_and_check_better(self, valid_losses):
        isbetter = False
        if len(self.valid_losses[0]) >= 1:
            for index in range(len(valid_losses)):
                cur_minN_valid = np.sort(self.valid_losses[index])[: self.num_avg][-1]
                if cur_minN_valid > valid_losses[index]:
                    isbetter = True
                    self.lr_descend_flag = 0
                    self.early_stop_flag = 0
            if not isbetter:
                self.lr_descend_flag += 1
                self.early_stop_flag += 1
        else:
            isbetter = True
        for index in range(len(valid_losses)):
            self.valid_losses[index].append(valid_losses[index])
        return isbetter

    def make_epoch_log(self, epoch, start, train_avg_loss, losses=None):
        log = "epoch:%d->%d|time: %.2f min|train_loss:%.4f" % (
            int(epoch),
            int(epoch + 1),
            (time.time() - start) / 60,
            train_avg_loss,
        )
        if losses is not None:
            for index in range(len(losses)):
                log += "|valid_loss_%d:%.4f;" % (index + 1, losses[index])
        print(log)

    def make_final_log(self):
        log = ""
        for index in range(len(self.valid_losses)):
            log += "loss%d best:%d(%.5f)" % (
                index + 1,
                np.argmin(self.valid_losses[index]).item() + 1,
                np.min(self.valid_losses[index]),
            )
        log += "stop_patience:%d/%d\n" % (self.early_stop_flag, self.stop_patience)
        print(log)

    def save_model(self, epoch):
        if self.local_rank == 0 or self.local_rank is None:
            checkpoint = {
                "model": self.ddp_model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch + 1,
                "valid_losses": self.valid_losses,
                "saved_models": self.saved_path,
            }
            model_save_path = os.path.join(
                self.save_home, "_ckpt_epoch_%d.ckpt" % (epoch + 1)
            )
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            torch.save(checkpoint, model_save_path)
            self.saved_path.append(model_save_path)
            print("saving checkpoint model to %s" % model_save_path)
