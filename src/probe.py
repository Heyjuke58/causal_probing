from torch.nn import CrossEntropyLoss, Dropout, LayerNorm, Linear, Module, Sequential, Tanh
from torch.optim import Adam


class TenneyMLP(Module):
    """2 layer MLP used by Tenney et al. in https://arxiv.org/abs/1905.06316."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, learning_rate):
        super().__init__()

        self.loss = CrossEntropyLoss()

        self.model = Sequential(
            Linear(input_dim, hidden_dim),
            Tanh(),
            LayerNorm(hidden_dim),
            Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, inputs):
        return self.model(inputs)
