"""
Cross-validation and retrospective evaluation entrypoint for the DPLM-2 HI-distance
DPO + antigenic map pipeline.

使用方式示例：
1) 交叉验证：
python dplm_hi_cv_retrospective_modified.py \
  --base-script "train_antigen_dplm_modified.py" \
  --eval-mode cv \
  --fold-num 5 \
  --cv-num 1 \
  --output-dir outputs_cv

2) 回顾性测试：
python dplm_hi_cv_retrospective.py \
  --base-script "Pasted code.py" \
  --eval-mode retrospective \
  --train-year-num 5 \
  --test-year-num 1 \
  --output-dir outputs_retro
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split

# 把另一个python文件导入，以作为一个模块，方便引其中的函数和变量
def import_base_module(base_script: Path):
    """Import the original training script even if its filename contains spaces."""
    base_script = base_script.resolve() # 获得绝对路径
    spec = importlib.util.spec_from_file_location("dplm_hi_base", base_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script: {base_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dplm_hi_base"] = module
    spec.loader.exec_module(module)
    return module


def infer_subtype_from_paths(*paths: Optional[Path]) -> str:
    text = " ".join(str(path).upper() for path in paths if path is not None)
    if "H3" in text:
        return "H3"
    if "H5" in text:
        return "H5"
    return "unknown"

def resolve_output_dir_by_subtype(args: argparse.Namespace) -> Path:
    subtype = args.subtype
    if subtype == "auto":
        subtype = infer_subtype_from_paths(args.ha, args.hi, args.structure_dir)
    args.subtype = subtype
    output_dir = args.output_dir
    if subtype != "unknown" and output_dir.name.upper() != subtype.upper():
        output_dir = output_dir / subtype
    args.output_dir = output_dir
    return output_dir


def add_pair_years(hi: pd.DataFrame, virus_records: Dict[int, object]) -> pd.DataFrame:
    """Add at_year/sr_year columns if the HI table does not already contain them."""
    hi = hi.copy()
    if "at_year" not in hi.columns:
        hi["at_year"] = hi["at_index"].map(lambda x: int(virus_records[int(x)].year))
        print(f'at_year has error')
        exit()
    if "sr_year" not in hi.columns:
        hi["sr_year"] = hi["sr_index"].map(lambda x: int(virus_records[int(x)].year))
        print(f'sr_year has error')
        exit()
    return hi


def unique_virus_indices(hi_part: pd.DataFrame) -> List[int]:
    indices = set(hi_part["at_index"].astype(int).tolist()) | set(hi_part["sr_index"].astype(int).tolist())
    return sorted(indices)


def subset_records(virus_records: Dict[int, object], allowed_indices: Iterable[int]) -> Dict[int, object]:
    allowed = set(map(int, allowed_indices))
    return {idx: record for idx, record in virus_records.items() if int(idx) in allowed}


def safe_sample_hi_triplets(base, args, hi_part: pd.DataFrame, seed: int):
    """Sample HI triplets; return an empty list instead of crashing on sparse folds."""
    if len(hi_part) == 0:
        return []
    try:
        return base.sample_hi_triplets(
            hi=hi_part,
            distance_threshold=args.distance_threshold,
            distance_scale=args.distance_scale,
            samples_per_anchor=args.hi_triplets_per_anchor,
            seed=seed,
            mode=args.hi_triplet_mode,
        )
    except ValueError as exc:
        print(f"[warning] no HI triplets sampled for this split: {exc}")
        return []


def sample_sequence_triplets_restricted(
    base,
    virus_records: Dict[int, object],
    anchors: Iterable[int],
    candidate_indices: Iterable[int],
    seq_threshold: float,
    seq_scale: float,
    samples_per_anchor: int,
    seed: int,
):
    """
    Same idea as the original sample_sequence_triplets, but positive/negative candidates
    are restricted to train+val viruses only, preventing test-virus leakage in stage2.
    """
    rng = np.random.default_rng(seed)
    candidate_indices = sorted(set(map(int, candidate_indices)))
    candidate_set = set(candidate_indices)
    triplets = []

    for anchor in anchors:
        anchor = int(anchor)
        if anchor not in candidate_set or anchor not in virus_records:
            continue

        candidates = []
        for candidate in candidate_indices:
            if candidate == anchor or candidate not in virus_records:
                continue
            diff = base.sequence_difference(virus_records[anchor].seq, virus_records[candidate].seq)
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
                base.TripletRecord(
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


@torch.no_grad()
def compute_coordinates_for_pairs(base, model, feature_store, hi_part: pd.DataFrame, device: torch.device) -> Dict[int, np.ndarray]:
    """Encode only viruses that appear in the given labeled pairs."""
    model.eval()
    coords: Dict[int, np.ndarray] = {}
    for virus_index in unique_virus_indices(hi_part):
        features = base.move_feature_batch(feature_store.get(int(virus_index)), device)
        point = model(**features).squeeze(0).detach().cpu().numpy().astype(float)
        coords[int(virus_index)] = point
    return coords


def evaluate_pairs(
    base,
    model,
    feature_store,
    hi_part: pd.DataFrame,
    device: torch.device,
    distance_min: float,
    distance_max: float,
    split_name: str,
    fold_name: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate labeled pairs by comparing Euclidean map distance with HI distance.
    The model was trained on normalized distances, so pred_norm is converted back
    to the raw distance scale before MAE/MSE/RMSE are computed.
    """
    if len(hi_part) == 0:
        return {
            "fold": fold_name,
            "split": split_name,
            "n_pairs": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "mae_norm": np.nan,
            "mse_norm": np.nan,
            "rmse_norm": np.nan,
            "acc": np.nan,
            "auc": np.nan,
            "auprc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "specificity": np.nan,
            "balanced_acc": np.nan,
            "threshold": 2.0,
        }

    coords = compute_coordinates_for_pairs(base, model, feature_store, hi_part, device)
    distance_range = max(float(distance_max) - float(distance_min), 1e-8)
    rows = []

    for row in hi_part.itertuples(index=False):
        a = int(row.at_index)
        b = int(row.sr_index)
        true_distance = float(row.distance)
        pred_norm = float(np.linalg.norm(coords[a] - coords[b]))
        pred_distance = pred_norm * distance_range + float(distance_min)
        true_norm = (true_distance - float(distance_min)) / distance_range
        rows.append(
            {
                "fold": fold_name,
                "split": split_name,
                "at_index": a,
                "sr_index": b,
                "at_year": int(getattr(row, "at_year")) if hasattr(row, "at_year") else np.nan,
                "sr_year": int(getattr(row, "sr_year")) if hasattr(row, "sr_year") else np.nan,
                "true_distance": true_distance,
                "pred_distance": pred_distance,
                "true_label": int(true_distance >= 2.0),
                "pred_label": int(pred_distance >= 2.0),
                "pred_score": pred_distance,
                "error": pred_distance - true_distance,
                "abs_error": abs(pred_distance - true_distance),
                "sq_error": (pred_distance - true_distance) ** 2,
                "true_distance_norm": true_norm,
                "pred_distance_norm": pred_norm,
                "error_norm": pred_norm - true_norm,
            }
        )

    pred_df = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_dir / f"{split_name}_pair_predictions.csv", index=False)

    err = pred_df["error"].to_numpy(dtype=float)
    err_norm = pred_df["error_norm"].to_numpy(dtype=float)

    pearson = pred_df[["pred_distance", "true_distance"]].corr(method="pearson").iloc[0, 1]
    spearman = pred_df[["pred_distance", "true_distance"]].corr(method="spearman").iloc[0, 1]

    y_true = pred_df["true_label"].to_numpy(dtype=int)
    y_pred = pred_df["pred_label"].to_numpy(dtype=int)
    y_score = pred_df["pred_score"].to_numpy(dtype=float)
    has_two_classes = len(np.unique(y_true)) == 2
    auc = roc_auc_score(y_true, y_score) if has_two_classes else np.nan
    auprc = average_precision_score(y_true, y_score) if has_two_classes else np.nan
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "fold": fold_name,
        "split": split_name,
        "n_pairs": int(len(pred_df)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(math.sqrt(np.mean(err ** 2))),
        "pearson": float(pearson) if not pd.isna(pearson) else np.nan,
        "spearman": float(spearman) if not pd.isna(spearman) else np.nan,
        "mae_norm": float(np.mean(np.abs(err_norm))),
        "mse_norm": float(np.mean(err_norm ** 2)),
        "rmse_norm": float(math.sqrt(np.mean(err_norm ** 2))),
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc) if not pd.isna(auc) else np.nan,
        "auprc": float(auprc) if not pd.isna(auprc) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)) if has_two_classes else np.nan,
        "threshold": 2.0,
    }


def save_split_indices(
    out_dir: Path,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    train_val_idx: Sequence[int],
    test_idx: Sequence[int],
    hi: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in [
        ("train_indices", train_idx),
        ("val_indices", val_idx),
        ("train_val_indices", train_val_idx),
        ("test_indices", test_idx),
    ]:
        pd.DataFrame({"hi_row_index": list(map(int, idx))}).to_csv(out_dir / f"{name}.csv", index=False)
    hi.iloc[list(map(int, train_idx))].to_csv(out_dir / "train_pairs.csv", index=False)
    hi.iloc[list(map(int, val_idx))].to_csv(out_dir / "val_pairs.csv", index=False)
    hi.iloc[list(map(int, train_val_idx))].to_csv(out_dir / "train_val_pairs.csv", index=False)
    hi.iloc[list(map(int, test_idx))].to_csv(out_dir / "test_pairs.csv", index=False)


def train_and_evaluate_one_split(
    base,
    args,
    hi: pd.DataFrame,
    virus_records: Dict[int, object],
    feature_store,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    fold_name: str,
) -> List[Dict[str, float]]:
    """Run stage1 + stage2 on train pairs; use val for early stopping and evaluate train/val/test separately."""
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    train_val_idx = np.unique(np.concatenate([train_idx, val_idx])).astype(int)

    split_dir = args.output_dir / fold_name
    save_split_indices(split_dir, train_idx, val_idx, train_val_idx, test_idx, hi)

    hi_train = hi.iloc[train_idx].reset_index(drop=True)
    hi_val = hi.iloc[val_idx].reset_index(drop=True)
    hi_train_val = hi.iloc[train_val_idx].reset_index(drop=True)
    hi_test = hi.iloc[test_idx].reset_index(drop=True)

    if len(hi_train_val) == 0:
        print(f"[skip] {fold_name}: empty train+val split")
        return []

    fold_args = copy.deepcopy(args)
    fold_args.output_dir = split_dir
    base.set_seed(args.seed)

    train_virus_indices = unique_virus_indices(hi_train)
    train_records = subset_records(virus_records, train_virus_indices)

    hi_triplets = safe_sample_hi_triplets(base, fold_args, hi_train, seed=args.seed)
    base.save_triplets(hi_triplets, split_dir / "hi_dpo_triplets_train.csv")
    val_hi_triplets = safe_sample_hi_triplets(base, fold_args, hi_val, seed=args.seed + 1000) if len(hi_val) > 0 else []
    base.save_triplets(val_hi_triplets, split_dir / "hi_dpo_triplets_val.csv")

    seq_triplets = sample_sequence_triplets_restricted(
        base=base,
        virus_records=train_records,
        anchors=base.build_distance_neighbors(hi_train).keys(),
        candidate_indices=train_virus_indices,
        seq_threshold=args.seq_threshold,
        seq_scale=args.seq_scale,
        samples_per_anchor=args.seq_triplets_per_anchor,
        seed=args.seed + 1,
    )
    base.save_triplets(seq_triplets, split_dir / "sequence_triplets_train.csv")

    device = torch.device(args.device)
    if args.skip_stage1 or len(hi_triplets) == 0:
        if len(hi_triplets) == 0 and not args.skip_stage1:
            print(f"[warning] {fold_name}: fallback to randomly initialized/pretrained backbone because no DPO triplets were available.")
        backbone = base.build_backbone(fold_args).to(device)
    else:
        backbone = base.train_stage1_dpo(
            fold_args,
            hi_triplets,
            feature_store,
            split_dir,
            device,
            val_triplets=val_hi_triplets,
        )

    model = base.train_stage2_map(
        fold_args,
        backbone,
        hi_train,
        seq_triplets,
        feature_store,
        virus_records,
        split_dir,
        device,
        val_hi=hi_val,
    )

    distance_min = float(hi_train["distance"].min())
    distance_max = float(hi_train["distance"].max())

    metrics = []
    for split_name, hi_part in [
        ("train", hi_train),
        ("val", hi_val),
        ("train_val", hi_train_val),
        ("test", hi_test),
    ]:
        metrics.append(
            evaluate_pairs(
                base=base,
                model=model,
                feature_store=feature_store,
                hi_part=hi_part,
                device=device,
                distance_min=distance_min,
                distance_max=distance_max,
                split_name=split_name,
                fold_name=fold_name,
                output_dir=split_dir,
            )
        )

    pd.DataFrame(metrics).to_csv(split_dir / "metrics.csv", index=False)

    del model, backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def make_cv_splits(hi: pd.DataFrame, fold_num: int, cv_num: int) -> Iterable[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Match the user's CV framework: KFold over HI row order. For each cv seed and fold,
    split train_val_idx into train/val using test_size=1/(fold_num-1), but train+val
    will be combined for the actual two-stage training.
    """
    all_rows = np.arange(len(hi))
    for cv in range(cv_num):
        kf = KFold(n_splits=fold_num, random_state=cv, shuffle=True)
        for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_rows)):
            train_idx, val_idx, _, _ = train_test_split(
                train_val_idx,
                train_val_idx,
                test_size=1 / (fold_num - 1),
                random_state=cv,
            )
            fold_name = f"cv{cv:02d}_fold{fold:02d}"
            yield fold_name, np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def make_retrospective_splits(args, hi: pd.DataFrame) -> Iterable[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Match the user's retrospective framework:
    - year_list = sorted(set(at_year))
    - train years are cumulative: year_list[0:cv + train_year_num]
    - test years are the following test_year_num years
    - train pair condition: at_year in train_year_list and sr_year < test_start_year
    - test pair condition: at_year in test_year_list; by default also sr_year < test_start_year
    Val is drawn from train_idx only for bookkeeping; train+val are then combined for training.
    """
    year_list = sorted(hi["at_year"].dropna().astype(int).unique().tolist())
    print("year_list:", year_list)
    cv_num = len(year_list) - args.train_year_num - args.test_year_num + 1
    if cv_num <= 0:
        raise ValueError(
            f"Not enough years for retrospective testing: len(year_list)={len(year_list)}, "
            f"train_year_num={args.train_year_num}, test_year_num={args.test_year_num}"
        )

    for cv in range(cv_num):
        train_year_list = year_list[0 : cv + args.train_year_num]
        test_year_list = year_list[cv + args.train_year_num : cv + args.train_year_num + args.test_year_num]
        test_start_year = year_list[cv + args.train_year_num]

        train_mask = hi["at_year"].isin(train_year_list) & (hi["sr_year"] < test_start_year)
        test_mask = hi["at_year"].isin(test_year_list) & (hi["sr_year"] < test_start_year)

        train_all_idx = hi.index[train_mask].to_numpy(dtype=int)
        test_idx = hi.index[test_mask].to_numpy(dtype=int)

        if len(train_all_idx) >= 2 and args.val_ratio > 0:
            train_idx, val_idx, _, _ = train_test_split(
                train_all_idx,
                train_all_idx,
                test_size=args.val_ratio,
                random_state=args.seed,
                # shuffle=True,
            )
        else:
            train_idx = train_all_idx
            val_idx = train_all_idx

        fold_name = f"retro{cv:02d}_train{train_year_list[0]}-{train_year_list[-1]}_test{'-'.join(map(str, test_year_list))}"
        yield fold_name, np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def parse_args() -> argparse.Namespace:
    """
    Parse wrapper-only arguments with parse_known_args(), then pass the remaining
    arguments to the original script's parse_args(). This lets you use all original
    arguments such as --ha/--hi/--encoder/--stage1-epochs unchanged.
    """
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--base-script", type=Path, default=Path("train_antigen_dplm_modified.py"))
    wrapper_parser.add_argument("--eval-mode", choices=["cv", "retrospective"], default="cv")
    wrapper_parser.add_argument("--subtype", choices=["auto", "H3", "H5", "unknown"], default="auto")
    wrapper_parser.add_argument("--fold-num", type=int, default=5)
    wrapper_parser.add_argument("--cv-num", type=int, default=5)
    wrapper_parser.add_argument("--train-year-num", type=int, default=5)
    wrapper_parser.add_argument("--test-year-num", type=int, default=1)
    wrapper_parser.add_argument("--val-ratio", type=float, default=0.2)

    wrapper_args, base_argv = wrapper_parser.parse_known_args()
    base = import_base_module(wrapper_args.base_script)

    # 把 wrapper 参数和 base 脚本参数合并到一个统一的 args 对象里
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + base_argv
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    for key, value in vars(wrapper_args).items():
        setattr(args, key, value)
    # 在 args 上加一个 _base_module 属性，存储 base 模块对象
    setattr(args, "_base_module", base)
    return args

def main() -> None:
    args = parse_args()
    base = args._base_module
    # 删掉args中的这个base这个模块
    delattr(args, "_base_module")

    # 随机种子
    base.set_seed(args.seed)
    # 识别亚型
    resolve_output_dir_by_subtype(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.ensure_struct_seq_fasta(args)

    # 把参数写入到文件中
    with open(args.output_dir / "config_cv_retrospective.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    virus_records = base.load_ha(
        args.ha,
        args.structure_dir,
        args.struct_seq_fasta,
        recursive_structure_search=not args.no_recursive_structure_search,
    )

    # 同一个序列用一个结构文件
    if not args.no_dedupe_structure_by_seq:
        structure_map = base.deduplicate_structures_by_sequence(virus_records)
        structure_map.to_csv(args.output_dir / "structure_sequence_dedup_mapping.csv", index=False)
    base.save_virus_input_mapping(virus_records, args.output_dir / "virus_input_mapping.csv")

    # 读出HI文件
    hi = base.load_hi(args.hi, virus_records, args.require_structure, require_struct_seq=args.encoder == "dplm2")
    hi = add_pair_years(hi, virus_records).reset_index(drop=True)
    hi.to_csv(args.output_dir / "all_labeled_pairs_with_years.csv", index=False)

    # 构建特征类
    feature_store = base.VirusFeatureStore(virus_records, args.max_seq_len, args.structure_dim)

    # 两种验证手段
    if args.eval_mode == "cv":
        split_iter = make_cv_splits(hi, fold_num=args.fold_num, cv_num=args.cv_num)
    else:
        split_iter = make_retrospective_splits(args, hi)

    # 
    all_metrics: List[Dict[str, float]] = []
    for fold_name, train_idx, val_idx, test_idx in split_iter:
        print(
            f"\n===== {fold_name} =====\n"
            f"n_train={len(train_idx)}, n_val={len(val_idx)}, n_train_val={len(np.unique(np.concatenate([train_idx, val_idx])))}, n_test={len(test_idx)}"
        )
        metrics = train_and_evaluate_one_split(
            base=base,
            args=args,
            hi=hi,
            virus_records=virus_records,
            feature_store=feature_store,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            fold_name=fold_name,
        )
        all_metrics.extend(metrics)
        if metrics:
            test_metrics = [m for m in metrics if m["split"] == "test"]
            if test_metrics:
                m = test_metrics[0]
                print(
                    f"[test metrics] {fold_name}: "
                    f"MAE={m['mae']:.6f}, MSE={m['mse']:.6f}, RMSE={m['rmse']:.6f}, "
                    f"Pearson={m['pearson']:.6f}, Spearman={m['spearman']:.6f}, "
                    f"ACC={m['acc']:.6f}, AUC={m['auc']:.6f}, AUPRC={m['auprc']:.6f}, F1={m['f1']:.6f}, n={m['n_pairs']}"
                )

    # 保存所有的指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.output_dir / "all_split_metrics.csv", index=False)

    # 输出指标
    if len(metrics_df) > 0:
        test_df = metrics_df[metrics_df["split"] == "test"].copy()
        if len(test_df) > 0:
            summary = test_df[["mae", "mse", "rmse", "pearson", "spearman", "mae_norm", "mse_norm", "rmse_norm", "acc", "auc", "auprc", "precision", "recall", "f1", "specificity", "balanced_acc"]].agg(["mean", "std"])
            summary.to_csv(args.output_dir / "test_metrics_summary.csv")
            print("\n===== Test metrics summary =====")
            print(summary)

    print(f"Done. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
