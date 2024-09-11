import numpy as np
import torch
import torch.nn as nn

from modules.spkadapt import SpeakerFuseLayer, SpeakerTransform


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.rnn = nn.LSTM(
            input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional
        )

        # linear projection layer
        self.proj = nn.Linear(
            hidden_size * 2, input_size
        )  # hidden_size = feature_dim * 2

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.norm(input).transpose(1, 2).contiguous())
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(input.shape[0], input.shape[2], input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


"""
TODO : attach the speaker embedding to each input
Input shape:(B,feature_dim + spk_emb_dim , T)
"""


class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7, bidirectional=True):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(
            self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional
        )
        self.band_comm = ResRNN(
            self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional
        )

    """
    permute() is meant to rearranges the dimension according to the provided order
    e.g. (A,B,C,D).permute(0,3,2,1) = (A,D,C,B)

    contiguous() is used to ensure that the memory layout of the tensor is contiguous
    """

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1)
        ).view(B, self.nband, -1, T)

        # band comm
        band_output = (
            band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.nband)
        )
        output = (
            self.band_comm(band_output)
            .view(B, T, -1, self.nband)
            .permute(0, 3, 2, 1)
            .contiguous()
        )

        return output.view(B, N, T)


class BSRNN(nn.Module):
    # self, sr=16000, win=512, stride=128, feature_dim=128, num_repeat=6, use_bidirectional=True
    def __init__(
        self,
        spk_emb_dim=256,
        sr=16000,
        win=512,
        stride=128,
        feature_dim=128,
        num_repeat=6,
        use_spk_transform=True,
        use_bidirectional=True,
        spk_fuse_type="concat",
    ):
        super(BSRNN, self).__init__()
        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.spk_emb_dim = spk_emb_dim

        # # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf

        # 0-8k (1k hop), 8k-16k (2k hop), 16k
        bandwidth_100 = int(np.floor(100 / (sr / 2.0) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.0) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.0) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.0) * self.enc_dim))

        # add up to 8k
        self.band_width = [bandwidth_100] * 15
        self.band_width += [bandwidth_200] * 10
        self.band_width += [bandwidth_500] * 5
        self.band_width += [bandwidth_2k] * 1

        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)

        if use_spk_transform:
            self.spk_transform = SpeakerTransform()
        else:
            self.spk_transform = nn.Identity()

        self.spk_fuse = SpeakerFuseLayer(
            embed_dim=self.spk_emb_dim,
            feat_dim=self.feature_dim,
            fuse_type=spk_fuse_type,
        )

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.nband * self.feature_dim, self.nband))
        self.separator = nn.Sequential(*self.separator)

        # self.proj =  nn.Linear(hidden_size*2, input_size)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * 4, 1),
                )
            )

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input, embeddings):
        # input shape: (B, C, T)
        # print(input.shape)
        # print(embeddings.shape)
        wav_input = input
        spk_emb_input = embeddings
        batch_size, nsample = wav_input.shape
        nch = 1

        # frequency-domain separation
        spec = torch.stft(
            wav_input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win)
            .to(wav_input.device)
            .type(wav_input.type()),
            return_complex=True,
        )

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_mix_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(
                spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )
            subband_mix_spec.append(
                spec[:, band_idx : band_idx + self.band_width[i]]
            )  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec[i].view(batch_size * nch, self.band_width[i] * 2, -1)
                )
            )
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        spk_embedding = self.spk_transform(spk_emb_input)
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)
        subband_feature = self.spk_fuse(subband_feature, spk_embedding)

        iput = subband_feature.view(batch_size * nch, self.nband * self.feature_dim, -1)
        sep_output = self.separator(iput)
        # B, nband*N, T
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.band_width[i], -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(
                this_output[:, 1]
            )  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = (
                subband_mix_spec[i].real * this_mask_real
                - subband_mix_spec[i].imag * this_mask_imag
            )  # B*nch, BW, T
            est_spec_imag = (
                subband_mix_spec[i].real * this_mask_imag
                + subband_mix_spec[i].imag * this_mask_real
            )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T
        output = torch.istft(
            est_spec.view(batch_size * nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win)
            .to(wav_input.device)
            .type(wav_input.type()),
            length=nsample,
        )

        output = output.view(batch_size, nch, -1)
        return torch.squeeze(output, dim=1)
