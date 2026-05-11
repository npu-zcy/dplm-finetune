import argparse
import hashlib
import importlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXBZJUO"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
PAD_ID = 0

DEFAULT_HA_PATH = Path("dataset/NHT/H5_NHT_HA.csv")
DEFAULT_HI_PATH = Path("dataset/NHT/H5_NHT_HI.csv")
DEFAULT_STRUCTURE_DIR = Path(
    "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
    "research2_geometric_graph_learning/myResearch/data/features/Structures/H5N1"
)
DEFAULT_DPLM_MODEL = Path(
    "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
    "research2_dplm/my_dplm/airkingbd/dplm2_150m"
)
DEFAULT_DPLM_STRUCT_TOKENIZER_DIR = Path(
    "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
    "research2_dplm/my_dplm/airkingbd/struct_tokenizer"
)


@dataclass
class VirusRecord:
    index: int
    name: str
    location: str
    virus_id: str
    year: int
    seq: str
    structure_path: Optional[Path]


@dataclass
class TripletRecord:
    anchor: int
    positive: int
    negative: int
    pos_distance: float
    neg_distance: float
    distance_gap: float
    margin: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ha(
    ha_path: Path,
    structure_dir: Optional[Path],
    structure_exts: Tuple[str, ...] = (".pt", ".npy", ".npz", ".pdb", ".cif"),
    recursive_structure_search: bool = True,
) -> Dict[int, VirusRecord]:
    ha = pd.read_csv(ha_path)
    required = {"index", "name", "location", "id", "year", "seq"}
    missing = required - set(ha.columns)
    if missing:
        raise ValueError(f"HA file is missing columns: {sorted(missing)}")

    structure_index = build_structure_file_index(structure_dir, structure_exts, recursive_structure_search)
    records: Dict[int, VirusRecord] = {}
    for row in ha.itertuples(index=False):
        virus_index = int(getattr(row, "index"))
        virus_id = str(getattr(row, "id"))
        structure_path = find_structure_file(structure_index, virus_id, str(getattr(row, "name")))
        records[virus_index] = VirusRecord(
            index=virus_index,
            name=str(getattr(row, "name")),
            location=str(getattr(row, "location")),
            virus_id=virus_id,
            year=int(getattr(row, "year")),
            seq=str(getattr(row, "seq")).strip().upper(),
            structure_path=structure_path,
        )
    return records


def build_structure_file_index(
    structure_dir: Optional[Path],
    exts: Tuple[str, ...],
    recursive: bool,
) -> Dict[str, Path]:
    if structure_dir is None or not structure_dir.exists():
        return {}

    paths = structure_dir.rglob("*") if recursive else structure_dir.glob("*")
    index: Dict[str, Path] = {}
    allowed_exts = {ext.lower() for ext in exts}
    for path in sorted(paths):
        if not path.is_file() or path.suffix.lower() not in allowed_exts:
            continue
        index.setdefault(path.stem, path)
        index.setdefault(sanitize_filename(path.stem), path)
    return index


def find_structure_file(structure_index: Dict[str, Path], virus_id: str, name: str) -> Optional[Path]:
    candidates = [virus_id, sanitize_filename(virus_id), name, sanitize_filename(name)]
    for stem in candidates:
        if stem in structure_index:
            return structure_index[stem]
    return None


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def short_sequence_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]


def deduplicate_structures_by_sequence(records: Dict[int, VirusRecord]) -> pd.DataFrame:
    seq_groups: Dict[str, List[VirusRecord]] = {}
    for record in records.values():
        seq_groups.setdefault(record.seq, []).append(record)

    rows = []
    for seq, group in seq_groups.items():
        group = sorted(group, key=lambda item: item.index)
        structure_records = [record for record in group if record.structure_path is not None]
        shared_path = structure_records[0].structure_path if structure_records else None
        representative_index = structure_records[0].index if structure_records else group[0].index
        representative_id = structure_records[0].virus_id if structure_records else group[0].virus_id

        for record in group:
            original_path = record.structure_path
            record.structure_path = shared_path
            rows.append(
                {
                    "index": record.index,
                    "id": record.virus_id,
                    "name": record.name,
                    "seq_hash": short_sequence_hash(seq),
                    "seq_length": len(seq),
                    "representative_index": representative_index,
                    "representative_id": representative_id,
                    "original_structure_path": str(original_path) if original_path is not None else "",
                    "shared_structure_path": str(shared_path) if shared_path is not None else "",
                    "shared_group_size": len(group),
                }
            )
    return pd.DataFrame(rows).sort_values(["seq_hash", "index"])


def load_hi(hi_path: Path, virus_records: Dict[int, VirusRecord], require_structure: bool) -> pd.DataFrame:
    hi = pd.read_csv(hi_path)
    required = {"at_index", "sr_index", "max_year", "min_year", "distance", "class"}
    missing = required - set(hi.columns)
    if missing:
        raise ValueError(f"HI file is missing columns: {sorted(missing)}")

    valid_indices = set(virus_records)
    hi = hi[hi["at_index"].isin(valid_indices) & hi["sr_index"].isin(valid_indices)].copy()
    hi["distance"] = pd.to_numeric(hi["distance"], errors="coerce")
    hi = hi.dropna(subset=["distance"])

    if require_structure:
        has_structure = {idx for idx, rec in virus_records.items() if rec.structure_path is not None}
        hi = hi[hi["at_index"].isin(has_structure) & hi["sr_index"].isin(has_structure)].copy()

    if hi.empty:
        raise ValueError("No usable HI rows remain after filtering.")
    return hi


def build_distance_neighbors(hi: pd.DataFrame) -> Dict[int, List[Tuple[int, float]]]:
    neighbors: Dict[int, Dict[int, float]] = {}
    for row in hi.itertuples(index=False):
        a = int(row.at_index)
        b = int(row.sr_index)
        d = float(row.distance)
        neighbors.setdefault(a, {})
        neighbors.setdefault(b, {})
        neighbors[a][b] = min(d, neighbors[a].get(b, d))
        neighbors[b][a] = min(d, neighbors[b].get(a, d))

    return {
        anchor: sorted(items.items(), key=lambda item: item[1])
        for anchor, items in neighbors.items()
        if len(items) >= 2
    }


def sample_hi_triplets(
    hi: pd.DataFrame,
    distance_threshold: float,
    distance_scale: float,
    samples_per_anchor: int,
    seed: int,
    mode: str,
) -> List[TripletRecord]:
    rng = np.random.default_rng(seed)
    neighbors = build_distance_neighbors(hi)
    triplets: List[TripletRecord] = []

    for anchor, candidates in neighbors.items():
        if len(candidates) < 2:
            continue

        mid = max(1, len(candidates) // 2)
        close_candidates = candidates[:mid]
        far_candidates = candidates[mid:]
        if not far_candidates:
            far_candidates = candidates[1:]

        if mode == "all":
            for pos_index, pos_distance in close_candidates:
                for neg_index, neg_distance in far_candidates:
                    if pos_index == neg_index:
                        continue

                    distance_gap = neg_distance - pos_distance
                    if distance_gap < distance_threshold:
                        continue

                    margin = distance_scale * distance_gap / max(distance_threshold, 1e-8)
                    triplets.append(
                        TripletRecord(
                            anchor=anchor,
                            positive=pos_index,
                            negative=neg_index,
                            pos_distance=pos_distance,
                            neg_distance=neg_distance,
                            distance_gap=distance_gap,
                            margin=margin,
                        )
                    )
            continue

        sampled = 0
        attempts = 0
        while sampled < samples_per_anchor and attempts < samples_per_anchor * 50:
            attempts += 1
            pos_index, pos_distance = close_candidates[int(rng.integers(0, len(close_candidates)))]
            neg_index, neg_distance = far_candidates[int(rng.integers(0, len(far_candidates)))]
            if pos_index == neg_index:
                continue

            distance_gap = neg_distance - pos_distance
            if distance_gap < distance_threshold:
                continue

            margin = distance_scale * distance_gap / max(distance_threshold, 1e-8)
            triplets.append(
                TripletRecord(
                    anchor=anchor,
                    positive=pos_index,
                    negative=neg_index,
                    pos_distance=pos_distance,
                    neg_distance=neg_distance,
                    distance_gap=distance_gap,
                    margin=margin,
                )
            )
            sampled += 1

    if not triplets:
        raise ValueError(
            "No HI triplets were sampled. Try lowering --distance-threshold or increasing HI coverage."
        )
    return triplets


def sequence_difference(seq_a: str, seq_b: str) -> float:
    max_len = max(len(seq_a), len(seq_b), 1)
    min_len = min(len(seq_a), len(seq_b))
    mismatches = sum(seq_a[i] != seq_b[i] for i in range(min_len))
    mismatches += max_len - min_len
    return mismatches / max_len


def sample_sequence_triplets(
    virus_records: Dict[int, VirusRecord],
    anchors: Iterable[int],
    seq_threshold: float,
    seq_scale: float,
    samples_per_anchor: int,
    seed: int,
) -> List[TripletRecord]:
    rng = np.random.default_rng(seed)
    indices = list(virus_records)
    triplets: List[TripletRecord] = []

    for anchor in anchors:
        candidates = []
        for candidate in indices:
            if candidate == anchor:
                continue
            diff = sequence_difference(virus_records[anchor].seq, virus_records[candidate].seq)
            candidates.append((candidate, diff))
        candidates.sort(key=lambda item: item[1])

        if len(candidates) < 2:
            continue

        mid = max(1, len(candidates) // 2)
        close_candidates = candidates[:mid]
        far_candidates = candidates[mid:]
        sampled = 0
        attempts = 0

        while sampled < samples_per_anchor and attempts < samples_per_anchor * 50:
            attempts += 1
            pos_index, pos_diff = close_candidates[int(rng.integers(0, len(close_candidates)))]
            neg_index, neg_diff = far_candidates[int(rng.integers(0, len(far_candidates)))]
            diff_gap = neg_diff - pos_diff
            if diff_gap < seq_threshold:
                continue

            margin = seq_scale * diff_gap / max(seq_threshold, 1e-8)
            triplets.append(
                TripletRecord(
                    anchor=anchor,
                    positive=pos_index,
                    negative=neg_index,
                    pos_distance=pos_diff,
                    neg_distance=neg_diff,
                    distance_gap=diff_gap,
                    margin=margin,
                )
            )
            sampled += 1

    return triplets


class StructureLoader:
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.cache: Dict[Path, torch.Tensor] = {}

    def load(self, path: Optional[Path]) -> torch.Tensor:
        if path is None:
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        if path in self.cache:
            return self.cache[path].clone()

        suffix = path.suffix.lower()
        if suffix == ".pt":
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("embedding", "features", "x", "coords"):
                    if key in obj:
                        obj = obj[key]
                        break
            tensor = torch.as_tensor(obj, dtype=torch.float32)
        elif suffix == ".npy":
            tensor = torch.as_tensor(np.load(path), dtype=torch.float32)
        elif suffix == ".npz":
            data = np.load(path)
            first_key = data.files[0]
            tensor = torch.as_tensor(data[first_key], dtype=torch.float32)
        else:
            # Raw pdb/cif parsing is model-specific. This zero vector lets the
            # data pipeline run; replace this branch with your structure encoder.
            tensor = torch.zeros(self.feature_dim, dtype=torch.float32)

        tensor = tensor.float().flatten()
        if tensor.numel() >= self.feature_dim:
            feature = tensor[: self.feature_dim]
            self.cache[path] = feature
            return feature.clone()

        padded = torch.zeros(self.feature_dim, dtype=torch.float32)
        padded[: tensor.numel()] = tensor
        self.cache[path] = padded
        return padded.clone()


def encode_sequence(seq: str, max_length: int) -> torch.Tensor:
    ids = [AA_TO_ID.get(aa, AA_TO_ID["X"]) for aa in seq[:max_length]]
    if len(ids) < max_length:
        ids += [PAD_ID] * (max_length - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class VirusFeatureStore:
    def __init__(self, records: Dict[int, VirusRecord], max_seq_len: int, structure_dim: int):
        self.records = records
        self.max_seq_len = max_seq_len
        self.structure_loader = StructureLoader(structure_dim)

    def get(self, virus_index: int) -> Dict[str, torch.Tensor]:
        record = self.records[int(virus_index)]
        return {
            "seq_ids": encode_sequence(record.seq, self.max_seq_len),
            "structure": self.structure_loader.load(record.structure_path),
            "index": torch.tensor(record.index, dtype=torch.long),
        }


class TripletDataset(Dataset):
    def __init__(self, triplets: List[TripletRecord], feature_store: VirusFeatureStore):
        self.triplets = triplets
        self.feature_store = feature_store

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        triplet = self.triplets[idx]
        return {
            "anchor": self.feature_store.get(triplet.anchor),
            "positive": self.feature_store.get(triplet.positive),
            "negative": self.feature_store.get(triplet.negative),
            "margin": torch.tensor(triplet.margin, dtype=torch.float32),
        }


class PairDataset(Dataset):
    def __init__(
        self,
        hi: pd.DataFrame,
        feature_store: VirusFeatureStore,
        distance_min: float,
        distance_max: float,
    ):
        self.hi = hi.reset_index(drop=True)
        self.feature_store = feature_store
        self.distance_min = distance_min
        self.distance_range = max(distance_max - distance_min, 1e-8)

    def __len__(self) -> int:
        return len(self.hi)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.hi.iloc[idx]
        distance = (float(row["distance"]) - self.distance_min) / self.distance_range
        return {
            "left": self.feature_store.get(int(row["at_index"])),
            "right": self.feature_store.get(int(row["sr_index"])),
            "distance": torch.tensor(distance, dtype=torch.float32),
        }


def move_feature_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class SimpleMultimodalEncoder(nn.Module):
    """Fallback encoder for debugging the whole pipeline before wiring dplm-2."""

    def __init__(self, vocab_size: int, structure_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
        self.seq_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.structure_proj = nn.Sequential(
            nn.Linear(structure_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, seq_ids: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        mask = seq_ids.ne(PAD_ID).float().unsqueeze(-1)
        seq_emb = self.embedding(seq_ids)
        pooled = (seq_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        seq_feature = self.seq_proj(pooled)
        structure_feature = self.structure_proj(structure)
        return self.fusion(torch.cat([seq_feature, structure_feature], dim=-1))


class DPLM2Backbone(nn.Module):
    """
    Thin adapter for your local dplm-2 multimodal model.

    Expected custom class signature:
        model = YourModel.from_pretrained(path_or_name) or YourModel(path_or_name)
        embedding = model(seq_ids=..., structure=...)

    If your dplm-2 API differs, edit only this class.
    """

    def __init__(
        self,
        module_name: str,
        class_name: str,
        model_name_or_path: str,
        struct_tokenizer_dir: Optional[str] = None,
    ):
        super().__init__()
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        model_name_or_path = str(model_name_or_path)
        struct_tokenizer_dir = str(struct_tokenizer_dir) if struct_tokenizer_dir else None
        if hasattr(cls, "from_pretrained"):
            cfg_override = {"tokenizer": {"vocab_file": model_name_or_path}}
            if struct_tokenizer_dir:
                cfg_override["struct_tokenizer"] = {"exp_path": struct_tokenizer_dir}
            self.model = cls.from_pretrained(model_name_or_path, cfg_override=cfg_override)
        else:
            self.model = cls(model_name_or_path)

    def forward(self, seq_ids: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        output = self.model(seq_ids=seq_ids, structure=structure)
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("pooler_output", "last_hidden_state", "embedding", "embeddings"):
                if key in output:
                    value = output[key]
                    return value.mean(dim=1) if value.dim() == 3 else value
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.mean(dim=1)
        raise TypeError("Unsupported dplm-2 output type. Please adapt DPLM2Backbone.forward().")


def maybe_apply_lora(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("peft is not installed. Install peft or run without --use-lora.")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, config)


def build_backbone(args: argparse.Namespace) -> nn.Module:
    if args.encoder == "simple":
        return SimpleMultimodalEncoder(
            vocab_size=len(AA_TO_ID) + 1,
            structure_dim=args.structure_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.embedding_dim,
        )

    if not args.dplm_module or not args.dplm_class or not args.dplm_model:
        raise ValueError("--dplm-module, --dplm-class and --dplm-model are required for --encoder dplm2")
    return DPLM2Backbone(args.dplm_module, args.dplm_class, args.dplm_model, args.dplm_struct_tokenizer_dir)


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return -torch.linalg.vector_norm(a - b, dim=-1)


def dpo_triplet_loss(
    policy_anchor: torch.Tensor,
    policy_positive: torch.Tensor,
    policy_negative: torch.Tensor,
    ref_anchor: torch.Tensor,
    ref_positive: torch.Tensor,
    ref_negative: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    policy_pref = similarity(policy_anchor, policy_positive) - similarity(policy_anchor, policy_negative)
    ref_pref = similarity(ref_anchor, ref_positive) - similarity(ref_anchor, ref_negative)
    return -F.logsigmoid(beta * (policy_pref - ref_pref)).mean()


def margin_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    pos_dist = torch.linalg.vector_norm(anchor - positive, dim=-1)
    neg_dist = torch.linalg.vector_norm(anchor - negative, dim=-1)
    return F.relu(pos_dist - neg_dist + margin).mean()


def encode_batch(model: nn.Module, features: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(seq_ids=features["seq_ids"], structure=features["structure"])


def train_stage1_dpo(
    args: argparse.Namespace,
    triplets: List[TripletRecord],
    feature_store: VirusFeatureStore,
    output_dir: Path,
    device: torch.device,
) -> nn.Module:
    policy = maybe_apply_lora(build_backbone(args), args).to(device)
    reference = build_backbone(args).to(device)
    reference.load_state_dict(policy.base_model.model.state_dict() if hasattr(policy, "base_model") else policy.state_dict(), strict=False)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    dataset = TripletDataset(triplets, feature_store)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    policy.train()
    for epoch in range(1, args.stage1_epochs + 1):
        losses = []
        for batch in loader:
            anchor = move_feature_batch(batch["anchor"], device)
            positive = move_feature_batch(batch["positive"], device)
            negative = move_feature_batch(batch["negative"], device)
            margin = batch["margin"].to(device)

            p_a = encode_batch(policy, anchor)
            p_p = encode_batch(policy, positive)
            p_n = encode_batch(policy, negative)

            with torch.no_grad():
                r_a = encode_batch(reference, anchor)
                r_p = encode_batch(reference, positive)
                r_n = encode_batch(reference, negative)

            loss_dpo = dpo_triplet_loss(p_a, p_p, p_n, r_a, r_p, r_n, args.beta)
            loss_triplet = margin_triplet_loss(p_a, p_p, p_n, margin)
            loss = loss_dpo + args.lambda_triplet * loss_triplet

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        print(f"[stage1] epoch={epoch} loss={np.mean(losses):.6f}")

    stage1_dir = output_dir / "stage1_dplm2_lora"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(policy, "save_pretrained"):
        policy.save_pretrained(stage1_dir)
    else:
        torch.save(policy.state_dict(), stage1_dir / "model.pt")
    return policy


class AntigenMapModel(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_dim: int, train_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embedding_dim // 2),
            nn.Linear(embedding_dim // 2, 2),
        )
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, seq_ids: torch.Tensor, structure: torch.Tensor, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if seq_ids.dim() == 1:
            seq_ids = seq_ids.unsqueeze(0)
        if structure.dim() == 1:
            structure = structure.unsqueeze(0)
        embedding = self.backbone(seq_ids=seq_ids, structure=structure)
        return self.projector(embedding)


def variance_regularizer(points: torch.Tensor, min_variance: float) -> torch.Tensor:
    variance = points.var(dim=0, unbiased=False).mean()
    return F.relu(torch.tensor(min_variance, device=points.device) - variance)


def train_stage2_map(
    args: argparse.Namespace,
    backbone: nn.Module,
    hi: pd.DataFrame,
    seq_triplets: List[TripletRecord],
    feature_store: VirusFeatureStore,
    virus_records: Dict[int, VirusRecord],
    output_dir: Path,
    device: torch.device,
) -> AntigenMapModel:
    distance_min = float(hi["distance"].min())
    distance_max = float(hi["distance"].max())
    pair_dataset = PairDataset(hi, feature_store, distance_min, distance_max)
    pair_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    seq_dataset = TripletDataset(seq_triplets, feature_store) if seq_triplets else None
    seq_loader = (
        iter(DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers))
        if seq_dataset is not None
        else None
    )

    model = AntigenMapModel(backbone, args.embedding_dim, args.train_backbone_stage2).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr_stage2, weight_decay=args.weight_decay)

    for epoch in range(1, args.stage2_epochs + 1):
        losses = []
        model.train()
        for batch in pair_loader:
            left = move_feature_batch(batch["left"], device)
            right = move_feature_batch(batch["right"], device)
            target_distance = batch["distance"].to(device)

            y_left = model(**left)
            y_right = model(**right)
            map_distance = torch.linalg.vector_norm(y_left - y_right, dim=-1)
            loss_hi = F.mse_loss(map_distance, target_distance)
            loss_reg = variance_regularizer(torch.cat([y_left, y_right], dim=0), args.min_map_variance)

            loss_seq = torch.tensor(0.0, device=device)
            if seq_loader is not None and args.lambda_seq > 0:
                try:
                    seq_batch = next(seq_loader)
                except StopIteration:
                    seq_loader = iter(DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers))
                    seq_batch = next(seq_loader)

                anchor = move_feature_batch(seq_batch["anchor"], device)
                positive = move_feature_batch(seq_batch["positive"], device)
                negative = move_feature_batch(seq_batch["negative"], device)
                margin = seq_batch["margin"].to(device)
                y_a = model(**anchor)
                y_p = model(**positive)
                y_n = model(**negative)
                loss_seq = margin_triplet_loss(y_a, y_p, y_n, margin)

            loss = loss_hi + args.lambda_seq * loss_seq + args.lambda_reg * loss_reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        print(f"[stage2] epoch={epoch} loss={np.mean(losses):.6f}")

    export_coordinates(model, feature_store, virus_records, output_dir, device)
    torch.save(
        {
            "model": model.state_dict(),
            "distance_min": distance_min,
            "distance_max": distance_max,
            "args": vars(args),
        },
        output_dir / "stage2_antigen_map.pt",
    )
    return model


@torch.no_grad()
def export_coordinates(
    model: AntigenMapModel,
    feature_store: VirusFeatureStore,
    virus_records: Dict[int, VirusRecord],
    output_dir: Path,
    device: torch.device,
) -> None:
    model.eval()
    rows = []
    for virus_index, record in virus_records.items():
        features = move_feature_batch(feature_store.get(virus_index), device)
        point = model(**features).squeeze(0).detach().cpu().numpy()
        rows.append(
            {
                "index": record.index,
                "name": record.name,
                "location": record.location,
                "id": record.virus_id,
                "year": record.year,
                "x": float(point[0]),
                "y": float(point[1]),
            }
        )
    pd.DataFrame(rows).sort_values("index").to_csv(output_dir / "antigen_map_coordinates.csv", index=False)


def save_triplets(triplets: List[TripletRecord], path: Path) -> None:
    pd.DataFrame([triplet.__dict__ for triplet in triplets]).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPLM-2 HI-distance DPO and antigen-map training pipeline.")
    parser.add_argument("--ha", type=Path, default=DEFAULT_HA_PATH, help="HA csv: index,name,location,id,year,seq")
    parser.add_argument("--hi", type=Path, default=DEFAULT_HI_PATH, help="HI csv: at_index,sr_index,max_year,min_year,distance,class")
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--no-recursive-structure-search", action="store_true")
    parser.add_argument("--no-dedupe-structure-by-seq", action="store_true")

    parser.add_argument("--encoder", choices=["simple", "dplm2"], default="dplm2")
    parser.add_argument("--dplm-module", default="byprot.models.dplm2")
    parser.add_argument("--dplm-class", default="MultimodalDiffusionProteinLanguageModel")
    parser.add_argument("--dplm-model", type=Path, default=DEFAULT_DPLM_MODEL)
    parser.add_argument("--dplm-struct-tokenizer-dir", type=Path, default=DEFAULT_DPLM_STRUCT_TOKENIZER_DIR)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--structure-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)

    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--use-lora", dest="use_lora", action="store_true")
    lora_group.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")

    parser.add_argument("--distance-threshold", type=float, default=1.0)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--hi-triplet-mode", choices=["sample", "all"], default="all")
    parser.add_argument("--hi-triplets-per-anchor", type=int, default=4)
    parser.add_argument("--seq-threshold", type=float, default=0.05)
    parser.add_argument("--seq-scale", type=float, default=1.0)
    parser.add_argument("--seq-triplets-per-anchor", type=int, default=2)

    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-stage2", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambda-triplet", type=float, default=0.5)
    parser.add_argument("--lambda-seq", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--min-map-variance", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-backbone-stage2", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    virus_records = load_ha(
        args.ha,
        args.structure_dir,
        recursive_structure_search=not args.no_recursive_structure_search,
    )
    if not args.no_dedupe_structure_by_seq:
        structure_map = deduplicate_structures_by_sequence(virus_records)
        structure_map.to_csv(output_dir / "structure_sequence_dedup_mapping.csv", index=False)

    hi = load_hi(args.hi, virus_records, args.require_structure)
    feature_store = VirusFeatureStore(virus_records, args.max_seq_len, args.structure_dim)

    hi_triplets = sample_hi_triplets(
        hi=hi,
        distance_threshold=args.distance_threshold,
        distance_scale=args.distance_scale,
        samples_per_anchor=args.hi_triplets_per_anchor,
        seed=args.seed,
        mode=args.hi_triplet_mode,
    )
    save_triplets(hi_triplets, output_dir / "hi_dpo_triplets.csv")

    seq_triplets = sample_sequence_triplets(
        virus_records=virus_records,
        anchors=build_distance_neighbors(hi).keys(),
        seq_threshold=args.seq_threshold,
        seq_scale=args.seq_scale,
        samples_per_anchor=args.seq_triplets_per_anchor,
        seed=args.seed + 1,
    )
    save_triplets(seq_triplets, output_dir / "sequence_triplets.csv")

    device = torch.device(args.device)
    if args.skip_stage1:
        backbone = build_backbone(args).to(device)
    else:
        backbone = train_stage1_dpo(args, hi_triplets, feature_store, output_dir, device)

    train_stage2_map(args, backbone, hi, seq_triplets, feature_store, virus_records, output_dir, device)
    print(f"Done. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
