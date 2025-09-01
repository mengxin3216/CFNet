import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url  # torch>=1.1
except Exception:
    load_state_dict_from_url = None


IMAGENET_VGG16_BN_URL = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"


class VGG16BN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        init_weights: bool = True,
        pretrained: bool = False,
        progress: bool = True,
    ):
        """
        直写版 VGG16-BN (配置D)

        Args:
            num_classes: 分类类别数
            dropout: classifier 的 Dropout 概率
            init_weights: 是否按官方方式初始化（当 pretrained=True 时会被自动关闭）
            pretrained: 是否加载 ImageNet 预训练权重
            progress: 下载预训练权重时是否显示进度
        """
        super().__init__()

        # ===== Backbone / features =====
        self.features = nn.Sequential(
            # block1: 64,64 + M
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block2: 128,128 + M
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block3: 256,256,256 + M
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block4: 512,512,512 + M
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block5: 512,512,512 + M
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        # 预训练权重时，禁用随机初始化
        if pretrained:
            init_weights = False

        if init_weights:
            self._initialize_weights()

        if pretrained:
            if load_state_dict_from_url is None:
                raise RuntimeError("torch.hub.load_state_dict_from_url 不可用，请升级 PyTorch 或手动加载权重。")
            state_dict = load_state_dict_from_url(IMAGENET_VGG16_BN_URL, progress=progress, check_hash=True)
            self.load_state_dict(state_dict, strict=False)  # 若你改了 num_classes，这里用 strict=False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _initialize_weights(module=None):
        """与 torchvision 官方一致的初始化：Conv=kaiming_normal, BN weight=1 bias=0, Linear=N(0,0.01)"""
        # 允许作为静态方法整体跑，也允许对单个 module 调用
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if module is None:
            # 对 self 递归
            # 在 __init__ 中调用时：self.apply(...)
            # 这里写成静态方法方便独立使用
            pass
        # 实际调用方式：在 __init__ 里 self.apply(_init)
        # 这里兼容两种用法：
        if module is None:
            # this branch won't be used; kept for clarity
            return
        else:
            module.apply(_init)
