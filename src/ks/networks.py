from functools import lru_cache

import torch
from torch import nn


class SingleHiddenLayerNN(nn.Module):
    def __init__(self, hidden_size: int, resolution: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.resolution = resolution
        self.shlp = nn.Sequential(
            nn.Linear(self.resolution, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.resolution),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shlp(x)


class ConcatSinCosNN(nn.Module):
    def __init__(self, hidden_size: int, resolution: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.resolution = resolution
        self.shlp = nn.Sequential(
            nn.Linear(3 * self.resolution, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.resolution),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        return self.shlp(x)


class CNNModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        linear_hidden_size: int,
        resolution: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.linear_hidden_size = linear_hidden_size
        self.resolution = resolution
        self.kernel_size = kernel_size

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.resolution * self.hidden_channels, self.linear_hidden_size),
            nn.ReLU(),
            nn.Linear(self.linear_hidden_size, self.linear_hidden_size),
            nn.ReLU(),
            nn.Linear(self.linear_hidden_size, self.resolution),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CircularCNN(nn.Module):
    def __init__(
        self,
        resolution: int,
        kernel_size: int = 3,
        depth: int = 5,
        num_channels: int = 16,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.depth = depth
        self.num_channels = num_channels

        layers = []
        for i in range(self.depth):
            in_channels = 1 if i == 0 else self.num_channels
            out_channels = 1 if i == self.depth - 1 else self.num_channels

            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding_mode="circular",
                    padding=self.kernel_size // 2,
                    bias=i != (self.depth - 1),
                )
            )

            if i < self.depth - 1:
                layers.append(nn.ReLU())

        self.deep_cnn = nn.Sequential(*layers)

    def apply_init(self, func, **kwargs):
        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                func(m.weight, **kwargs)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deep_cnn(x)


@lru_cache
def fourier_features(
    domain_size: float,
    resolution: int,
    frequencies: tuple[float, ...],
    device: torch.device,
) -> torch.Tensor:
    grid = torch.linspace(0, domain_size, resolution)
    features = []
    # TODO: make periodic
    for f in frequencies:
        features.append(torch.sin(2 * torch.pi * f * grid))
        features.append(torch.cos(2 * torch.pi * f * grid))
    return torch.stack(features, dim=-1).to(device)


class CircularCNNDomainSizes(nn.Module):
    def __init__(
        self,
        resolution: int,
        device: torch.device,
        kernel_size: int = 3,
        depth: int = 5,
        num_channels: int = 16,
        num_extra_in_channels: int = 0,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.device = device
        self.kernel_size = kernel_size
        self.depth = depth
        self.num_channels = num_channels

        layers = []
        for i in range(self.depth):
            in_channels = (1 + num_extra_in_channels) if i == 0 else self.num_channels
            out_channels = 1 if i == self.depth - 1 else self.num_channels

            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding_mode="circular",
                    padding=self.kernel_size // 2,
                    bias=i != (self.depth - 1),
                    device=self.device,
                )
            )

            if i < self.depth - 1:
                layers.append(nn.ReLU())

        self.deep_cnn = nn.Sequential(*layers)

    def apply_init(self, func, **kwargs):
        def init_weights(m):
            if isinstance(m, nn.Conv1d):
                func(m.weight, **kwargs)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deep_cnn(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        device: torch.device,
        kernel_size=3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_ch,
            out_channels=hidden_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
            device=device,
        )
        self.bn1 = nn.BatchNorm1d(hidden_ch, device=device)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
            device=device,
        )
        self.bn2 = nn.BatchNorm1d(out_ch, device=device)

        self.shortcut = (
            nn.Conv1d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=1, device=device
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        resolution: int,
        device: torch.device,
        kernel_size: int = 3,
        depth: int = 5,
        num_channels: int = 16,
        num_extra_in_channels: int = 0,
    ):
        super().__init__()
        self.resolution = resolution
        self.device = device
        self.kernel_size = kernel_size
        self.depth = depth
        self.num_channels = num_channels

        layers = []

        for i in range(self.depth):
            layers.append(
                ResidualBlock(
                    in_ch=(1 + num_extra_in_channels) if i == 0 else num_channels,
                    hidden_ch=num_channels,
                    out_ch=num_channels if i < self.depth - 1 else 1,
                    kernel_size=self.kernel_size,
                    device=self.device,
                )
            )

        self.dnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnn(x)


class ResidualBlockV2(nn.Module):
    def __init__(self, channels, kernel_size=3, padding_mode="circular", use_bias=True):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
            bias=use_bias,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
            bias=use_bias,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out


class ResNet1DV2(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        res_net_channels,
        res_net_depth,
        kernel_size=3,
        padding_mode="circular",
        use_bias=True,
    ):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Conv1d(
                input_channels,
                res_net_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
                bias=use_bias,
            ),
            nn.BatchNorm1d(res_net_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlockV2(res_net_channels, kernel_size, padding_mode, use_bias)
                for _ in range(res_net_depth)
            ]
        )

        self.downsample = nn.Sequential(
            nn.Conv1d(
                res_net_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
                bias=use_bias,
            ),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.residual_blocks(x)
        x = self.downsample(x)
        return x
