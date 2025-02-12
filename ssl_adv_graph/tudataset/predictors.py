from torch import nn

class PPI_MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.
    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.
    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class ZINC_MLP_Predictor(nn.Module):
    r"""MLP used for predictor in a regression setting.
    Args:
        input_size (int): Size of input features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`512`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        # Set output size to 1 for regression
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)  # Output size is 1 for ZINC regression
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # Kaiming uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
