import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=63,
        hidden_dim=128,
        output_dim=5,
        drop_rate=0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)

    def save(self, path, extra: dict | None = None) -> None:
        """Save the weights (and optionally extra metadata) to ``path``."""
        payload = {"state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location="cpu", **kwargs) -> "MLP":
        """Rebuild a model from a checkpoint written by :meth:`save`."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

