import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HybridYearLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        annual_metric_dim: int,
        num_classes: int,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        seq_embed_dim: int = 128,
        annual_embed_dim: int = 64,
        hybrid_embed_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_sequence_branch: bool = True,
        use_annual_metrics: bool = True,
    ):
        super().__init__()

        self.use_sequence_branch = use_sequence_branch
        self.use_annual_metrics = use_annual_metrics
        self.bidirectional = bidirectional

        active_dims = []

        if self.use_sequence_branch:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            lstm_base_dim = lstm_hidden_dim * (2 if bidirectional else 1)

            self.attn = nn.Sequential(
                nn.Linear(lstm_base_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

            pooled_dim = lstm_base_dim * 3

            self.seq_proj = nn.Sequential(
                nn.Linear(pooled_dim, seq_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            active_dims.append(seq_embed_dim)

        if self.use_annual_metrics:
            if annual_metric_dim <= 0:
                raise ValueError("USE_ANNUAL_METRICS=True but annual_metric_dim is 0")
            self.annual_mlp = nn.Sequential(
                nn.Linear(annual_metric_dim, annual_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            active_dims.append(annual_embed_dim)

        if len(active_dims) == 0:
            raise ValueError("At least one branch must be enabled.")

        combined_dim = sum(active_dims)

        self.hybrid_proj = nn.Sequential(
            nn.Linear(combined_dim, hybrid_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hybrid_embed_dim, num_classes)

    def encode_sequence(self, x, lengths):
        packed = pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None].to(out.device)

        mask_f = mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(dim=1) / lengths.to(out.device).unsqueeze(1).float()

        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool, _ = out_masked.max(dim=1)

        attn_logits = self.attn(out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_pool = (out * attn_weights.unsqueeze(-1)).sum(dim=1)

        z_seq = torch.cat([attn_pool, mean_pool, max_pool], dim=1)
        return self.seq_proj(z_seq)

    def encode(self, x, lengths, annual_metrics):
        parts = []

        if self.use_sequence_branch:
            parts.append(self.encode_sequence(x, lengths))

        if self.use_annual_metrics:
            parts.append(self.annual_mlp(annual_metrics))

        z = torch.cat(parts, dim=1)
        hybrid_emb = self.hybrid_proj(z)
        return hybrid_emb

    def forward(self, x, lengths, annual_metrics):
        hybrid_emb = self.encode(x, lengths, annual_metrics)
        logits = self.classifier(hybrid_emb)
        return logits, hybrid_emb


