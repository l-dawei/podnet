import mindspore.nn as nn


class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
        super(AlexNet, self).__init__()

        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0

        self.features = nn.SequentialCell(
            nn.Conv2d(channel, 64, kernel_size=11, stride=4, pad_mode='same', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),

            nn.Conv2d(64, 128, kernel_size=5, pad_mode='same', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),

            nn.Conv2d(128, 192, kernel_size=3, pad_mode='same', has_bias=True),
            nn.ReLU(),

            nn.Conv2d(192, 256, kernel_size=3, pad_mode='same', has_bias=True),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, pad_mode='same', has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
        )
        self.classifier = nn.SequentialCell(
            nn.Flatten(),

            nn.Dense(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),

            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),

            nn.Dense(4096, num_classes)
        )

    def construct(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
