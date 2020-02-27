import torch.nn as nn


class PiPPNP(nn.Module):
    """The PiPPNP model. First run an MLP on the node features
    and then diffuse the logits using the Personalized PageRank Matrix.
    """
    def __init__(self, n_features, n_classes, n_hidden):
        super().__init__()

        self.linear = nn.ModuleList([nn.Linear(n_features, n_hidden[0])])
        self.linear.extend([nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(0, len(n_hidden) - 1)])
        self.linear.append(nn.Linear(n_hidden[-1], n_classes))

        self.act_func = nn.ReLU()

    def forward(self, attr, ppr):
        x = attr
        for layer in self.linear[:-1]:
            x = self.act_func(layer(x))

        logits = self.linear[-1](x)
        diffused_logits = ppr @ logits

        return logits, diffused_logits
