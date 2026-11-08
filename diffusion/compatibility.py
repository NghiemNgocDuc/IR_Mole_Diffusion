"""Experimental frozen-IR / molecular-fingerprint compatibility head.

Scores describe a learned association, not a forward IR calculation or a
probability of molecular correctness. Candidate scoring never uses references.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diffusion.ranking import canonical_identity


FINGERPRINT_BITS = 2048


def molecular_fingerprints(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS,
                                                          includeChirality=False)
    identities = [canonical_identity(value) for value in smiles]
    fingerprints = np.empty((len(identities), FINGERPRINT_BITS), dtype=np.uint8)
    for index, identity in enumerate(identities):
        fingerprints[index] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(identity))
    return torch.from_numpy(fingerprints), identities


def training_fingerprints(data, rows, allowed_training_rows):
    """Read molecular labels only for the exact, verified training selection."""
    if list(rows) != list(allowed_training_rows) or len(set(rows)) != len(rows):
        raise ValueError("Compatibility training rows must exactly match the training split.")
    if len(rows) < 2 or any(type(row) is not int or not 0 <= row < len(data["molecule_ids"]) for row in rows):
        raise ValueError("At least two valid integer training rows are required.")
    return molecular_fingerprints([str(data["molecule_ids"][row]) for row in rows])


class CompatibilityHead(nn.Module):
    def __init__(self, ir_dim, projection_dim=128):
        super().__init__()
        if min(ir_dim, projection_dim) < 1:
            raise ValueError("Projection dimensions must be positive.")
        self.spec = {"ir_dim": ir_dim, "projection_dim": projection_dim}
        def projection(input_dim):
            return nn.Sequential(nn.Linear(input_dim, projection_dim), nn.LayerNorm(projection_dim),
                                 nn.SiLU(), nn.Linear(projection_dim, projection_dim))
        self.ir_projection = projection(ir_dim)
        self.molecule_projection = projection(FINGERPRINT_BITS)

    def forward(self, spectra, fingerprints):
        ir = F.normalize(self.ir_projection(spectra.float()), dim=-1)
        molecules = F.normalize(self.molecule_projection(fingerprints.float()), dim=-1)
        return ir @ molecules.T


def identity_contrastive_loss(scores, identities, temperature=0.1):
    """Symmetric multi-positive InfoNCE; same identity is never a negative."""
    if not 0 < temperature or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive.")
    if scores.ndim != 2 or scores.shape != (len(identities), len(identities)) or not identities:
        raise ValueError("Training requires a square matrix of paired scores.")
    if not torch.isfinite(scores).all():
        raise ValueError("Compatibility scores must be finite.")
    positives = torch.tensor([[left == right for right in identities] for left in identities],
                             device=scores.device, dtype=torch.bool)
    logits = scores / temperature
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    return 0.5 * ((torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
                  + (torch.logsumexp(logits, dim=0) - torch.logsumexp(positive_logits, dim=0)).mean())


@torch.inference_mode()
def cache_spectrum_features(encoder, data, rows, device, batch_size=256, progress=None, label="Caching IR features"):
    """Masked mean-pool the frozen EMA encoder; read no molecule labels."""
    encoder.eval().requires_grad_(False)
    if batch_size < 1 or not rows:
        raise ValueError("Feature extraction needs rows and a positive batch size.")
    features = []
    if progress is not None:
        progress.start(label, len(rows), "spectra")
    for start in range(0, len(rows), batch_size):
        indices = rows[start:start + batch_size]
        spectra = torch.as_tensor(data["spectra"][indices], device=device)
        if data["packed"]:
            spectra = {"packed": spectra, "mask": torch.as_tensor(data["spectra_mask"][indices], device=device)}
        tokens, mask = encoder.encode_spectra(spectra)
        pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if not torch.isfinite(pooled).all():
            raise ValueError("Encoder returned nonfinite features.")
        features.append(pooled.cpu())
        if progress is not None:
            progress.update(start + len(indices))
    # Clone outside inference mode in the caller before using features in autograd.
    return torch.cat(features)


def fit_head(features, fingerprints, identities, *, device="cpu", epochs=5, batch_size=256,
             seed=78438379, lr=1e-4, projection_dim=128, progress=None):
    if epochs < 1 or batch_size < 2 or seed < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid compatibility training budget, batch size, seed, or learning rate.")
    if len(features) != len(fingerprints) or len(identities) != len(features) or len(set(identities)) < 2:
        raise ValueError("Compatibility pairs require matching rows and at least two distinct identities.")
    features, fingerprints = features.clone(), fingerprints.clone()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = CompatibilityHead(features.shape[1], projection_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        batches = list(torch.randperm(len(features), generator=generator).split(batch_size))
        if len(batches) > 1 and len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = torch.cat((batches[-1], last))
        if progress is not None:
            progress.start(f"Compatibility training epoch {epoch + 1}/{epochs}", len(features), "pairs")
        total_loss, count = 0.0, 0
        head.train()
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            scores = head(features[indices].to(device), fingerprints[indices].to(device))
            loss = identity_contrastive_loss(scores, [identities[index] for index in indices.tolist()])
            loss.backward()
            norm = nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            count += len(indices)
            total_loss += float(loss.detach()) * len(indices)
            if progress is not None:
                progress.update(count)
        history.append({"epoch": epoch + 1, "training_loss": total_loss / count, "training_rows": count,
                        "optimizer_steps": len(batches), "last_gradient_norm": float(norm)})
        print(f"Compatibility epoch {epoch + 1}/{epochs}: training loss {total_loss / count:.6f}", flush=True)
    return head.eval(), history


@torch.inference_mode()
def rerank_candidates(head, ir_feature, candidates, device="cpu"):
    """Score only input IR and candidate structures; tie-break by original order."""
    if not candidates:
        return []
    fingerprints, _ = molecular_fingerprints([candidate["smiles"] for candidate in candidates])
    head.eval()
    scores = head(ir_feature.reshape(1, -1).to(device), fingerprints.to(device))[0].cpu().tolist()
    if not np.isfinite(scores).all():
        raise ValueError("Reranking produced nonfinite scores.")
    ordered = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    return [{**copy.deepcopy(candidates[index]), "decoder_rank": index + 1, "compatibility_score": scores[index]}
            for index in ordered]
