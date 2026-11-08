"""IR-to-SELFIES transformer with a self-contained vocabulary and encoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from diffusion.encoder import SpectrumTokenEncoder
from diffusion.model import PortableModule


class SpectrumStructureModel(PortableModule):
    def __init__(self, vocabulary, encoder_spec=None, decoder_layers=4, heads=4,
                 max_length=128, dropout=0.1, lr=2e-4, label_smoothing=0.05,
                 ema_decay=0.999, weight_decay=1e-5, ranking_weight=0.0, ranking_margin=0.1,
                 ranking_length_penalty=0.6):
        super().__init__()
        encoder_spec = dict(encoder_spec or {"hidden_dim": 256, "n_layers": 4})
        encoder_spec.pop("kind", None)
        if vocabulary[:3] != ["<pad>", "<bos>", "<eos>"] or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary must start with unique pad/bos/eos tokens.")
        if not 0 <= ema_decay < 1 or not 0 <= label_smoothing < 1 or max_length < 3:
            raise ValueError("Invalid EMA, smoothing, or maximum length.")
        if not all(math.isfinite(value) and value >= 0 for value in (ranking_weight, ranking_margin, ranking_length_penalty)):
            raise ValueError("Ranking weight and margin must be finite and nonnegative.")
        self.save_hyperparameters()
        self.encoder = SpectrumTokenEncoder(**encoder_spec)
        dim = self.encoder.spec["hidden_dim"]
        self.embedding = nn.Embedding(len(vocabulary), dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)
        block = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(block, decoder_layers, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, len(vocabulary))

    def decode(self, tokens, memory, memory_mask):
        length = tokens.shape[1]
        if length > self.hparams.max_length:
            raise ValueError("Sequence exceeds checkpoint maximum length.")
        positions = torch.arange(length, device=tokens.device)
        features = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim) + self.position(positions)
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        decoded = self.decoder(features, memory, tgt_mask=causal,
            tgt_key_padding_mask=tokens.eq(0), memory_key_padding_mask=~memory_mask)
        return self.output(decoded)

    def forward(self, spectra, tokens):
        memory, mask = self.encoder.encode_spectra(spectra)
        return self.decode(tokens, memory, mask)

    def step(self, batch, stage):
        target = batch["selfies_tokens"].long()
        memory, memory_mask = self.encoder.encode_spectra(batch["spectra"])
        logits = self.decode(target[:, :-1], memory, memory_mask)
        labels = target[:, 1:]
        token_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=0,
            reduction="none", label_smoothing=self.hparams.label_smoothing if stage == "train" else 0.)
        live = labels.ne(0)
        loss = (token_loss.sum(1) / live.sum(1).clamp_min(1)).mean()
        if stage == "train" and self.hparams.ranking_weight > 0:
            if "negative_selfies_tokens" not in batch:
                raise ValueError("Ranking training requires verified training-only negative SELFIES.")
            negative = batch["negative_selfies_tokens"].long()
            negative_logits = self.decode(negative[:, :-1], memory, memory_mask)
            negative_labels = negative[:, 1:]
            negative_live = negative_labels.ne(0)
            # Length-normalized conditional log-likelihood; negative labels are
            # alternate training identities, not tokens to suppress everywhere.
            positive_nll = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=0, reduction="none")
            negative_nll = F.cross_entropy(negative_logits.transpose(1, 2), negative_labels, ignore_index=0, reduction="none")
            # Include BOS in the length, as generate() does for completed beams.
            positive_score = -positive_nll.sum(1) / (live.sum(1) + 1).float().pow(self.hparams.ranking_length_penalty)
            negative_score = -negative_nll.sum(1) / (negative_live.sum(1) + 1).float().pow(self.hparams.ranking_length_penalty)
            ranking_loss = F.relu(self.hparams.ranking_margin - positive_score + negative_score).mean()
            loss = loss + self.hparams.ranking_weight * ranking_loss
            self.log("train_ranking_loss", ranking_loss, batch_size=len(target), prog_bar=True)
            self.log("train_pairwise_margin_accuracy", (positive_score > negative_score).float().mean(), batch_size=len(target))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite spectrum-to-structure loss.")
        accuracy = ((logits.argmax(-1).eq(labels) & live).sum(1) / live.sum(1)).mean()
        self.log(f"{stage}_loss", loss, batch_size=len(target), sync_dist=True, prog_bar=True)
        self.log(f"{stage}_token_accuracy", accuracy, batch_size=len(target), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    @torch.no_grad()
    def generate(self, spectra, beam_size=5, length_penalty=0.6, score_mode="legacy"):
        """Decode one spectrum; optionally use the final score during pruning too.

        Legacy remains the default for existing checkpoints and callers. Length
        includes BOS and EOS, matching the training pairwise score. Consistent
        scoring is an experiment, not a guarantee of a better candidate list.
        """
        if beam_size < 1 or beam_size > 100:
            raise ValueError("beam_size must be between 1 and 100.")
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError("length_penalty must be finite and nonnegative.")
        if score_mode not in ("legacy", "consistent"):
            raise ValueError("score_mode must be legacy or consistent.")
        memory, mask = self.encoder.encode_spectra(spectra)
        if len(memory) != 1:
            raise ValueError("Beam generation currently accepts one spectrum at a time.")
        beams = [([1], 0.)]
        completed = []
        for _ in range(self.hparams.max_length - 1):
            candidates = []
            for sequence, score in beams:
                tokens = torch.tensor([sequence], device=memory.device)
                logp = self.decode(tokens, memory, mask)[0, -1].float().log_softmax(-1)
                logp[:2] = -torch.inf
                if len(sequence) == 1:
                    logp[2] = -torch.inf
                values, indices = logp.topk(min(beam_size, len(logp)-2))
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidate = (sequence + [index], score + value)
                    if index == 2:
                        completed.append(candidate)
                    else:
                        candidates.append(candidate)
            if score_mode == "legacy":
                rank = lambda item: item[1] / (((5 + len(item[0])) / 6) ** length_penalty)
            else:
                rank = lambda item: item[1] / len(item[0])**length_penalty
            beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
            completed = sorted(completed, key=rank, reverse=True)[:beam_size]
            if not beams:
                break
        vocabulary = self.hparams.vocabulary
        results = []
        for tokens, score in sorted(completed, key=lambda item: item[1] / len(item[0])**length_penalty, reverse=True):
            results.append({"selfies": "".join(vocabulary[i] for i in tokens[1:-1]),
                            "log_probability": score})
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                "monitor": "val_loss", "strict": False}}
