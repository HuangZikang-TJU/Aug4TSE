import torch
from torch.nn.modules.loss import _Loss

class Power(_Loss):
    """
        Shape:
        - est_targets : :math:`(batch, time)`.
        - targets: :math:`(batch, time)`.
    """
    def __init__(self, zero_mean=True, EPS=1e-8):
        super(Power, self).__init__()
        self.zero_mean = zero_mean
        self.EPS = EPS

    def forward(self, est_target):
        if self.zero_mean:
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            est_target = est_target - mean_estimate

        ratio = torch.sum(est_target ** 2, axis=-1)
        sdr = 10 * torch.log10(ratio / est_target.shape[-1] * 16000 + self.EPS)
        return sdr

class SisdrPowerWrapper(_Loss):
    def __init__(self, sisdr_loss, power_loss, loss_weights, zero_mean=True):
        super(SisdrPowerWrapper, self).__init__()
        self.sisdr_loss = sisdr_loss
        self.power_loss = power_loss
        self.zero_mean = zero_mean
        self.loss_weights = loss_weights

    def forward(self, est_target, target, case_masks):
        if target.ndim == 3:
            target = target.squeeze(0)
        if est_target.ndim == 3:
            est_target = est_target.squeeze(0)
        if case_masks.ndim == 3:
            case_masks = case_masks.squeeze(0)
        if target.size() != est_target.size():
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {target.size()} and {est_target.size()} instead"
            )
        if self.zero_mean:
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            est_target = est_target - mean_estimate

        B, _ = target.shape
        est_target_speech = est_target[case_masks]
        target_speech = target[case_masks]
        # pack as batches, drop last few frames
        drop_frame = est_target_speech.shape[-1] // B * B
        est_target_speech = est_target_speech[:drop_frame].reshape(B, -1)
        target_speech = target_speech[:drop_frame].reshape(B, -1)

        # sisdr
        loss = self.loss_weights[0] * torch.mean(self.sisdr_loss(est_target_speech, target_speech))
        # loss = torch.mean(self.sisdr_loss(est_target[case_masks].unsqueeze(0), target[case_masks].unsqueeze(0)))

        # power
        est_target_speech = est_target[~case_masks]
        drop_frame = est_target_speech.shape[-1] // B * B
        est_target_speech = est_target_speech[:drop_frame].reshape(B, -1)
        loss += self.loss_weights[1] * torch.mean(self.power_loss(est_target_speech))
        return loss

power_loss = Power()

