import chainer
import chainer.functions as F


class LinkLeakyRelu(chainer.Chain):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.leaky_relu(x)
