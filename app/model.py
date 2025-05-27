from torch import Tensor, nn, flatten


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        if in_dim != out_dim:
            stride = 2
        self.conv = nn.Sequential(nn.BatchNorm2d(in_dim), nn.ReLU(),
                                  nn.Conv2d(in_dim, out_dim, 3, stride, 1),
                                  nn.BatchNorm2d(out_dim), nn.ReLU(),
                                  nn.Conv2d(out_dim, out_dim, 3, 1, 1))

        downsample = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, stride),
                                   nn.BatchNorm2d(out_dim))
        self.downsample = downsample if stride != 1 else lambda x: x

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.downsample(x)


class ResNet(nn.Module):
    def __init__(self, blocks: list) -> None:
        super().__init__()
        self.blocks = blocks
        self.init_block = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1))

        self._make_blocks([64, 128, 256, 512])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_layer = nn.Linear(12800, 1000)
        self.classify = nn.Linear(1000, 5)

    def _make_blocks(self, dimensions: list) -> None:
        inx, in_dim = 0, 64
        for block, out_dim in zip(self.blocks, dimensions):
            for _ in range(block):
                setattr(self, f'block{inx}', ResBlock(in_dim, out_dim))
                inx, in_dim = inx+1, out_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.init_block(x)

        for inx in range(sum(self.blocks)):
            x = getattr(self, f'block{inx}')(x)

        x = flatten(self.avgpool(x), 1)
        x = self.fc_layer(x)
        x = self.classify(x)

        return x

    def stage_inference(self) -> None:
        self.classify = nn.Sequential()
        self.eval()
