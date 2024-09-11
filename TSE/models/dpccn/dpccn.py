import torch
import torch.nn as nn
from modules.spkadapt import SpeakerFuseLayer, SpeakerTransform
from modules.convs import Conv2dBlock, ConvTrans2dBlock, DenseBlock, TCNBlock


class DPCCN(nn.Module):
    def __init__(
        self,
        win=512,
        stride=128,
        spk_emb_dim=256,
        use_spk_transform=False,
        spk_fuse_type="multiply",
        feature_dim=257,
        kernel_size=(3, 3),
        stride1=(1, 1),
        stride2=(1, 2),
        paddings=(1, 1),
        output_padding=(0, 0),
        tcn_dims=384,
        tcn_blocks=10,
        tcn_layers=2,
        causal=False,
        pool_size=(4, 8, 16, 32),
    ) -> None:
        super(DPCCN, self).__init__()

        self.win_len = win
        self.hop_size = stride
        self.spk_emb_dim = spk_emb_dim

        self.conv2d = nn.Conv2d(2, 16, kernel_size, stride1, paddings)

        self.encoder = self._build_encoder(
            kernel_size=kernel_size, stride=stride2, padding=paddings
        )

        if use_spk_transform:
            self.spk_transform = SpeakerTransform()
        else:
            self.spk_transform = nn.Identity()

        self.spk_fuse = SpeakerFuseLayer(
            embed_dim=self.spk_emb_dim,
            feat_dim=feature_dim,
            fuse_type=spk_fuse_type,
        )

        self.tcn_layers = self._build_tcn_layers(
            tcn_layers,
            tcn_blocks,
            in_dims=tcn_dims,
            out_dims=tcn_dims,
            causal=causal,
        )
        self.decoder = self._build_decoder(
            kernel_size=kernel_size,
            stride=stride2,
            padding=paddings,
            output_padding=output_padding,
        )
        self.avg_pool = self._build_avg_pool(pool_size)
        self.avg_proj = nn.Conv2d(64, 32, 1, 1)

        self.deconv2d = nn.ConvTranspose2d(32, 2, kernel_size, stride1, paddings)

    def _build_encoder(self, **enc_kargs):
        """
        Build encoder layers
        """
        encoder = nn.ModuleList()
        encoder.append(DenseBlock(16, 16, "enc"))
        for i in range(4):
            encoder.append(
                nn.Sequential(
                    Conv2dBlock(in_dims=16 if i == 0 else 32, out_dims=32, **enc_kargs),
                    DenseBlock(32, 32, "enc"),
                )
            )
        encoder.append(Conv2dBlock(in_dims=32, out_dims=64, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=64, out_dims=128, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=128, out_dims=384, **enc_kargs))

        return encoder

    def _build_decoder(self, **dec_kargs):
        """
        Build decoder layers
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384 * 2, out_dims=128, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=128 * 2, out_dims=64, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=64 * 2, out_dims=32, **dec_kargs))
        for i in range(4):
            decoder.append(
                nn.Sequential(
                    DenseBlock(32, 64, "dec"),
                    ConvTrans2dBlock(
                        in_dims=64, out_dims=32 if i != 3 else 16, **dec_kargs
                    ),
                )
            )
        decoder.append(DenseBlock(16, 32, "dec"))

        return decoder

    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        """
        Build TCN blocks in each repeat (layer)
        """
        blocks = [TCNBlock(**tcn_kargs, dilation=(2**b)) for b in range(tcn_blocks)]

        return nn.Sequential(*blocks)

    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        """
        Build TCN layers
        """
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs) for _ in range(tcn_layers)
        ]

        return nn.Sequential(*layers)

    def _build_avg_pool(self, pool_size):
        """
        Build avg pooling layers
        """
        avg_pool = nn.ModuleList()
        for sz in pool_size:
            avg_pool.append(nn.Sequential(nn.AvgPool2d(sz), nn.Conv2d(32, 8, 1, 1)))

        return avg_pool

    def forward(self, input, aux):
        wav_input = input
        spk_emb_input = aux
        batch_size, nsample = wav_input.shape

        # frequency-domain separation
        spec = torch.stft(
            wav_input,
            n_fft=self.win_len,
            hop_length=self.hop_size,
            window=torch.hann_window(self.win_len)
            .to(wav_input.device)
            .type(wav_input.type()),
            return_complex=True,
        )
        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)

        # spec = torch.einsum("hijk->hikj", spec_RI)  # batchsize, 2, T, F
        spec = torch.transpose(spec_RI, 2, 3)  # batchsize, 2, T, F
        out = self.conv2d(spec)
        out_list = []
        out = self.encoder[0](out)

        spk_embedding = self.spk_transform(spk_emb_input)
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)

        out = self.spk_fuse(out.transpose(2, 3), spk_embedding).transpose(2, 3)
        out_list.append(out)
        for _, enc in enumerate(self.encoder[1:]):
            out = enc(out)
            out_list.append(out)

        B, N, T, F = out.shape
        out = out.reshape(B, N, T * F)
        out = self.tcn_layers(out)
        out = out.reshape(B, N, T, F)
        out_list = out_list[::-1]
        for idx, dec in enumerate(self.decoder):
            out = dec(torch.cat([out_list[idx], out], 1))
            # Pyramidal pooling
        B, N, T, F = out.shape
        upsample = nn.Upsample(size=(T, F), mode="bilinear")
        pool_list = []
        for avg in self.avg_pool:
            pool_list.append(upsample(avg(out)))
        out = torch.cat([out, *pool_list], 1)
        out = self.avg_proj(out)

        out = self.deconv2d(out)

        est_spec = torch.transpose(out, 2, 3)  # (batchsize, 2, F, T)
        B, N, F, T = est_spec.shape
        est_spec = torch.chunk(est_spec, 2, 1)  # [(B, 1, F, T), (B, 1, F, T)])
        est_spec = torch.complex(est_spec[0], est_spec[1])

        output = torch.istft(
            est_spec.reshape(B, -1, T),
            n_fft=self.win_len,
            hop_length=self.hop_size,
            window=torch.hann_window(self.win_len)
            .to(wav_input.device)
            .type(wav_input.type()),
            length=nsample,
        )

        return output


if __name__ == "__main__":
    import numpy as np

    model = DPCCN()
    s = 0
    for param in model.parameters():
        s += np.product(param.size())
    print("# of parameters: " + str(s / 1024.0 / 1024.0))
    mix = torch.randn(4, 32000)
    aux = torch.randn(4, 256)
    est = model(mix, aux)
    print(est.size())
