import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def _safe_tensor(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(t):
        return t
    if torch.isnan(t).any() or torch.isinf(t).any():
        return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    return t


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding with dynamic extension."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            new_pe = self._build_pe(seq_len, self.d_model).to(self.pe.device).type_as(self.pe)
            self.pe = new_pe
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, embed_size: int, padding_idx: int = 0):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)


class PeakMLPEncoderV2(nn.Module):
    """Per-peak MLP encoder for IR peaklist input.
    Each peak is represented as a fixed-size vector (freq, intensity [, fwhm])
    and independently mapped to hidden_dim via an MLP.
    """

    def __init__(self, hidden_dim: int, n_layers: int = 4, max_peaks: int = 81,
                 in_channels: int = 2, peak_norm: str = 'layernorm_hidden',
                 **kwargs):
        super().__init__()
        self.max_peaks = max_peaks
        self.in_channels = in_channels

        layers = [nn.Linear(in_channels, hidden_dim), nn.SiLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

        if peak_norm == 'layernorm_hidden':
            self.norm = nn.LayerNorm(hidden_dim)
        elif peak_norm == 'batchnorm_token':
            self.norm = nn.BatchNorm1d(max_peaks)
        else:
            raise ValueError(f"Unknown peak_norm={peak_norm!r}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.net(x)
        h = self.norm(h)
        return h


class SpectraEncoder(nn.Module):
    """CNN encoder for dense (grid) IR spectra.
    Used when encoder_type='cnn'.  Three Conv1d layers with configurable
    strides produce a downsampled sequence of hidden_dim features.
    """

    def __init__(self, input_dim, hidden_dim, in_channels=1,
                 strides=(2, 2, 2)):
        super().__init__()
        s1, s2, s3 = strides
        self.conv1 = nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=7, stride=s1, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=s2, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=s3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.conv3(x)))
        return x


class BaseModel(pl.LightningModule):
    """Base class with formula prediction head and training/eval loops.
    The formula head predicts molecular formula (atom counts for
    C, H, N, O, F, P, S, Cl, Br, I) from a pooled encoder representation.
    """

    def __init__(self, formula_loss_weight: float = 0.5,
                 formula_weight_schedule: str = 'fixed',
                 formula_warmup_epochs: int = 20,
                 formula_initial_weight: float = 0.0,
                 formula_loss_normalize: bool = True,
                 label_smoothing: float = 0.0,
                 **kwargs):
        super().__init__()
        self.formula_loss_weight = formula_loss_weight
        self.formula_weight_schedule = formula_weight_schedule
        self.formula_warmup_epochs = formula_warmup_epochs
        self.formula_initial_weight = formula_initial_weight
        self.formula_loss_normalize = formula_loss_normalize
        self.label_smoothing = label_smoothing

    def _sanitize_input(self, x):
        if x is not None and x.dim() == 3 and x.size(-1) == 1:
            return x.squeeze(-1)
        return x

    def _process_spectra(self, spectra):
        """Encode input to [B, L, H] tokens + optional padding mask.
        Handles both peaklist dicts and dense tensors.
        """
        if isinstance(spectra, dict):
            peaks = spectra.get('packed', spectra.get('peaks'))
            mask = spectra.get('mask')
            peaks = _safe_tensor(peaks)
            enc = self.feature_extractor(peaks)
            pad_mask = ~mask.bool() if mask is not None else None
            return enc, pad_mask

        spectra = _safe_tensor(spectra)
        token_first = getattr(self, '_feature_extractor_is_token_first', False)
        if token_first:
            return self.feature_extractor(spectra), None

        if spectra.dim() == 2:
            spectra = spectra.unsqueeze(1)
        features = self.feature_extractor(spectra)
        return features.transpose(1, 2), None

    # Formula loss 

    def get_formula_loss_weight(self) -> float:
        target = self.formula_loss_weight
        if self.formula_weight_schedule == 'fixed' or target == 0.0:
            return target
        epoch = self.current_epoch
        warmup = self.formula_warmup_epochs
        initial = self.formula_initial_weight
        if epoch >= warmup:
            return target
        progress = epoch / max(warmup, 1)
        if self.formula_weight_schedule == 'linear_warmup':
            return initial + (target - initial) * progress
        if self.formula_weight_schedule == 'cosine_warmup':
            cos_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
            return initial + (target - initial) * cos_progress
        return target

    def _formula_loss(self, formula_pred, atom_counts):
        if self.formula_loss_normalize:
            pred_log = torch.log1p(torch.relu(formula_pred))
            true_log = torch.log1p(atom_counts)
            return F.mse_loss(pred_log, true_log)
        return F.mse_loss(formula_pred, atom_counts)

    # Sequence loss with optional label smoothing 

    def _seq_loss(self, log_probs, targets):
        ignore_idx = self.hparams.padding_idx
        mask = (targets != ignore_idx).float()
        denom = mask.sum().clamp(min=1.0)
        log_probs_fp32 = log_probs.float()
        nll_elem = F.nll_loss(
            log_probs_fp32, targets, ignore_index=ignore_idx, reduction='none',
        )
        nll_raw = (nll_elem * mask).sum() / denom
        if self.label_smoothing <= 0.0:
            zero = torch.zeros((), device=log_probs.device, dtype=nll_raw.dtype)
            return nll_raw, nll_raw, zero
        clamped = log_probs_fp32.clamp(min=-50.0)
        smooth_elem = -clamped.mean(dim=1)
        smooth_penalty = (smooth_elem * mask).sum() / denom
        s = float(self.label_smoothing)
        loss = (1.0 - s) * nll_raw + s * smooth_penalty
        return loss, nll_raw, smooth_penalty 

    def configure_optimizers(self):
        lr = self.hparams.lr if hasattr(self.hparams, 'lr') else 1e-4
        wd = self.hparams.weight_decay if hasattr(self.hparams, 'weight_decay') else 1e-5
        optimizer = optim.AdamW(
            self.parameters(), lr=lr, weight_decay=wd,
            betas=(0.9, 0.98), eps=1e-9,
        )
        total_steps = (
            self.trainer.estimated_stepping_batches
            if self.trainer.estimated_stepping_batches else 100000
        )
        warmup_steps = int(total_steps * 0.06)
        try:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps, num_cycles=0.5,
            )
        except ImportError:
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps,
                pct_start=0.06, anneal_strategy='cos',
                div_factor=25.0, final_div_factor=1e4,
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1},
        }


    def common_step(self, batch, batch_idx, stage='train'):
        if len(batch) >= 7:
            spectra, inputs, targets, atom_counts = batch[0], batch[1], batch[2], batch[3]
        elif len(batch) >= 6:
            spectra, inputs, targets, atom_counts = batch[0], batch[1], batch[2], batch[3]
        else:
            spectra, inputs, targets, atom_counts = batch
        inputs = self._sanitize_input(inputs)

        if not isinstance(spectra, dict):
            if torch.isnan(spectra).any() or torch.isinf(spectra).any():
                spectra = torch.nan_to_num(spectra, nan=0.0, posinf=1.0, neginf=-1.0)
        if (targets < 0).any() or (targets >= self.num_classes).any():
            targets = torch.clamp(targets, 0, self.num_classes - 1)

        logits, formula_pred = self(spectra, inputs)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return None

        logits_transposed = logits.transpose(1, 2)
        loss_seq, nll_raw, smooth_penalty = self._seq_loss(logits_transposed, targets)
        loss_formula = self._formula_loss(formula_pred, atom_counts)
        formula_weight = self.get_formula_loss_weight()
        total_loss = loss_seq + formula_weight * loss_formula

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return None

        on_s = (stage == 'train')
        self.log(f'{stage}_loss', total_loss, on_step=on_s, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_loss_seq', loss_seq, on_step=on_s, on_epoch=True)
        self.log(f'{stage}_loss_formula', loss_formula, on_step=on_s, on_epoch=True)

        if stage == 'val':
            pred_rounded = torch.round(torch.relu(formula_pred))
            exact = torch.all(torch.eq(pred_rounded, atom_counts), dim=1).float()
            self.log('val_formula_acc', exact.mean(), on_epoch=True, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, stage='train')
        if loss is None:
            return 0.0 * sum(p.sum() for p in self.parameters() if p.requires_grad)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, stage='val')
        if loss is None:
            sentinel = torch.tensor(1e6, device=self.device)
            self.log('val_loss', sentinel, on_step=False, on_epoch=True, prog_bar=True)
            return torch.tensor(0.0, device=self.device)
        return loss

    def predict_step(self, batch, batch_idx):
        spectra, inputs = batch[0], batch[1]
        batch_size = (spectra.shape[0] if not isinstance(spectra, dict)
                      else next(iter(spectra.values())).shape[0])
        inp = torch.zeros((batch_size, self.max_word_len), dtype=torch.long, device=self.device)
        if inputs.size(1) > 0:
            inp[:, 0] = inputs[:, 0]
        outs = torch.empty((batch_size, self.max_word_len, self.num_classes), device=self.device)
        for i in range(self.max_word_len - 1):
            with torch.no_grad():
                curr_out, _ = self(spectra, inp)
                outs[:, i] = curr_out[:, i]
                next_token = curr_out[:, i].argmax(dim=-1)
                if i + 1 < self.max_word_len:
                    inp[:, i + 1] = next_token
        return outs


class Transformer(BaseModel):
    def __init__(
        self,
        hidden_dim, dropout, layers, max_word_len, num_classes, input_dim,
        padding_idx, heads,
        input_channels=1, batch_size=64, lr=1e-4,
        weight_decay=1e-5, max_len=None, name='Transformer',
        formula_loss_weight=0.5, formula_weight_schedule='fixed',
        formula_warmup_epochs=20, formula_initial_weight=0.0,
        formula_loss_normalize=True, label_smoothing=0.0,
        encoder_type='cnn', cnn_strides=(2, 2, 2),
        peak_max_peaks=81, peak_mlp_layers=4,
        ir_rep='interp', ir_max_peaks=81,
        ir_peak_freq_scale=None, ir_peak_int_scale=None,
        # Decoder depth
        decoder_layers=None,
        peak_channels=2,
        mode='ir',
        peak_norm='layernorm_hidden',
        **kwargs,
    ):
        super().__init__(
            formula_loss_weight, formula_weight_schedule,
            formula_warmup_epochs, formula_initial_weight,
            formula_loss_normalize, label_smoothing,
        )
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.mode = mode

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})."
            )

        _decoder_layers = decoder_layers if decoder_layers is not None else layers

        # Feature extractor 
        if encoder_type == 'peak_mlp':
            self.feature_extractor = PeakMLPEncoderV2(
                hidden_dim, n_layers=peak_mlp_layers,
                max_peaks=peak_max_peaks, in_channels=peak_channels,
                peak_norm=peak_norm,
            )
            self._feature_extractor_is_token_first = True
        else:
            self.feature_extractor = SpectraEncoder(
                input_dim, hidden_dim, in_channels=input_channels,
                strides=tuple(cnn_strides),
            )
            self._feature_extractor_is_token_first = False

        self.input_projection = nn.Linear(hidden_dim, hidden_dim)

        # Formula prediction head: pooled encoder → 10 atom counts
        self.formula_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=heads,
            num_encoder_layers=layers,
            num_decoder_layers=_decoder_layers,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation='gelu',
        )

        # Decoder embeddings
        self.tgt_tok_emb = TokenEmbedding(
            self.num_classes, hidden_dim, padding_idx=padding_idx,
        )
        self.positional_encoding = PositionalEncoding(
            hidden_dim, dropout=dropout, max_len=max_len if max_len else 5000,
        )
        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()
        self.transformer.enable_nested_tensor = False

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, spectra, decoder_in):

        decoder_in = self._sanitize_input(decoder_in)

        # Encode spectrum
        spectra_enc, src_pad_mask = self._process_spectra(spectra)
        encoder_input = self.input_projection(spectra_enc)
        encoder_input = self.positional_encoding(encoder_input)
        memory = self.transformer.encoder(
            encoder_input, src_key_padding_mask=src_pad_mask,
        )

        # Pooled representation for formula prediction
        if src_pad_mask is not None:
            valid = (~src_pad_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            global_repr = (memory * valid).sum(dim=1) / denom
        else:
            global_repr = torch.mean(memory, dim=1)
        formula_pred = self.formula_head(global_repr)

        # Decode
        tgt_mask = self._generate_square_subsequent_mask(
            decoder_in.size(1), device=decoder_in.device,
        )
        tgt_padding_mask = (decoder_in == self.padding_idx)
        decoder_emb = self.tgt_tok_emb(decoder_in)
        decoder_emb = self.positional_encoding(decoder_emb)

        if self.training:
            decoder_emb = decoder_emb.clamp(-10, 10)

        output = self.transformer.decoder(
            tgt=decoder_emb, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        output = self.dropout_layer(output)
        output = output.clamp(-20, 20)
        logits = F.log_softmax(self.linear(output), dim=-1)
        return logits, formula_pred

    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
