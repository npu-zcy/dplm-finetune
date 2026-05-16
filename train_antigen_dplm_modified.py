import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
from tqdm import tqdm
import tempfile
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
ID_TO_AA = {idx: aa for aa, idx in AA_TO_ID.items()}
PAD_ID = 0

DEFAULT_HA_PATH = Path("dataset/NHT/H5_NHT_HA.csv")
DEFAULT_HI_PATH = Path("dataset/NHT/H5_NHT_HI.csv")
# DEFAULT_HA_PATH = Path("dataset/NHT/H3_NHT_full_HA.csv")
# DEFAULT_HI_PATH = Path("dataset/NHT/H3_NHT_full_HI.csv")
DEFAULT_STRUCTURE_DIR = Path(
    "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
    "research2_geometric_graph_learning/myResearch/data/features/Structures/H5N1"
)
# DEFAULT_STRUCTURE_DIR = Path(
#     "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
#     "research2_geometric_graph_learning/myResearch/data/features/Structures/H3N2"
# )
DEFAULT_DPLM_MODEL = Path(
    "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
    "research2_dplm/my_dplm/airkingbd/dplm2_150m"
)
DEFAULT_DPLM_STRUCT_TOKENIZER_DIR = Path(
    "/home/zhouchunyan/postgraduate/influenza-virus_LLM/"
    "research2_dplm/my_dplm/airkingbd/struct_tokenizer"
)

# 结构token
DEFAULT_STRUCT_SEQ_FASTA = Path("data/tokenized_protein/H5/struct_seq.fasta")
# DEFAULT_STRUCT_SEQ_FASTA = Path("data/tokenized_protein/H3/struct_seq.fasta")
DEFAULT_TOKENIZED_PROTEIN_DIR = Path("data/tokenized_protein")


def infer_subtype_from_paths(*paths: Optional[Path]) -> str:
    """Infer influenza subtype from input paths; falls back to "unknown"."""
    text = " ".join(str(path).upper() for path in paths if path is not None)
    if "H3" in text:
        return "H3"
    if "H5" in text:
        return "H5"
    return "unknown"


def resolve_output_dir_by_subtype(args: argparse.Namespace) -> Path:
    """Put H3/H5 runs into separate output folders unless the folder already ends with subtype."""
    subtype = args.subtype
    if subtype == "auto":
        subtype = infer_subtype_from_paths(args.ha, args.hi, args.structure_dir)
    args.subtype = subtype
    output_dir = args.output_dir
    if subtype != "unknown" and output_dir.name.upper() != subtype.upper():
        output_dir = output_dir / subtype
    args.output_dir = output_dir
    return output_dir


# 表示单个病毒的所有信息
@dataclass
class VirusRecord:
    index: int
    name: str
    location: str
    virus_id: str
    year: int
    seq: str
    structure_path: Optional[Path]
    struct_seq: Optional[str] = None

# 用于训练 DPO 或 triplet loss 的三元组
@dataclass
class TripletRecord:
    anchor: int
    positive: int
    negative: int
    pos_distance: float
    neg_distance: float
    distance_gap: float
    margin: float

# 
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 确保 DPLM-2 模型可以拿到每条病毒蛋白的结构 token 序列文件 struct_seq.fasta，如果没有就用 PDB 文件自动生成它，并返回这个路径。
def ensure_struct_seq_fasta(args: argparse.Namespace) -> Optional[Path]:
    # 如果用的是不需要结构/已经有对结构tokenizer后文件，则直接返回
    if args.encoder != "dplm2":
        return args.struct_seq_fasta
    if args.struct_seq_fasta is not None and args.struct_seq_fasta.exists():
        return args.struct_seq_fasta

    # 找 PDB tokenizer 脚本
    script_path = resolve_pdb_tokenizer_script(args.pdb_tokenizer_script)
    if script_path is None:
        raise FileNotFoundError(
            "struct_seq_fasta does not exist and no PDB tokenizer script was found. "
            "Please pass --pdb-tokenizer-script /path/to/dplm/src/byprot/utils/protein/tokenize_pdb.py "
            "or pre-generate --struct-seq-fasta."
        )

    # 是否存在结构文件
    if args.structure_dir is None or not args.structure_dir.exists():
        raise FileNotFoundError(f"Cannot tokenize PDB files because --structure-dir does not exist: {args.structure_dir}")

    output_dir = args.struct_seq_fasta.parent if args.struct_seq_fasta is not None else DEFAULT_TOKENIZED_PROTEIN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tokenize] struct_seq_fasta not found, generating with {script_path}")
    print(f"[tokenize] input_pdb_folder={args.structure_dir}")
    print(f"[tokenize] output_dir={output_dir}")

    # 如果你指定了本地 struct_tokenizer_dir，它会修改脚本，把 tokenizer 指向本地目录。
    tokenizer_script = make_local_tokenizer_script(script_path, args.dplm_struct_tokenizer_dir)

    # 设置离线环境并执行, 准备环境变量，确保 离线执行（不下载 HF 模型）。
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    cmd = [
        sys.executable,
        str(tokenizer_script),
        "--input_pdb_folder",
        str(args.structure_dir),
        "--output_dir",
        str(output_dir),
    ]
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if tokenizer_script != script_path and tokenizer_script.exists():
            tokenizer_script.unlink()


    generated = output_dir / "struct_seq.fasta"
    if not generated.exists():
        candidates = [output_dir / "struct.fasta"] + sorted(output_dir.glob("*struct*.fasta")) + sorted(output_dir.glob("*struct*.fa"))
        existing_candidates = [candidate for candidate in candidates if candidate.exists()]
        if existing_candidates:
            generated = existing_candidates[0]
        else:
            raise FileNotFoundError(
                f"PDB tokenizer finished but no struct_seq.fasta was found in {output_dir}."
            )
    args.struct_seq_fasta = generated
    print(f"[tokenize] using generated struct_seq_fasta={generated}")
    return generated

# 让 tokenize_pdb.py 使用你本地的结构 tokenizer：
def make_local_tokenizer_script(script_path: Path, struct_tokenizer_dir: Optional[Path]) -> Path:
    if struct_tokenizer_dir is None:
        return script_path
    if not struct_tokenizer_dir.exists():
        raise FileNotFoundError(
            f"--dplm-struct-tokenizer-dir does not exist: {struct_tokenizer_dir}"
        )

    text = script_path.read_text(encoding="utf-8")
    local_path = repr(str(struct_tokenizer_dir))
    replacements = [
        ("get_struct_tokenizer()", f"get_struct_tokenizer({local_path})"),
        ('get_struct_tokenizer("airkingbd/struct_tokenizer")', f"get_struct_tokenizer({local_path})"),
        ("get_struct_tokenizer('airkingbd/struct_tokenizer')", f"get_struct_tokenizer({local_path})"),
    ]
    patched = text
    changed = False
    for old, new in replacements:
        if old in patched:
            patched = patched.replace(old, new)
            changed = True

    if not changed:
        print(
            "[tokenize] warning: could not patch get_struct_tokenizer call; "
            "running original tokenizer script."
        )
        return script_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_tokenize_pdb_local.py",
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        tmp.write(patched)
    patched_path = Path(tmp.name)
    print(f"[tokenize] using local struct tokenizer: {struct_tokenizer_dir}")
    return patched_path


def resolve_pdb_tokenizer_script(script_arg: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if script_arg is not None:
        candidates.append(script_arg)
    candidates.extend(
        [
            Path("src/byprot/utils/protein/tokenize_pdb.py"),
            Path("../src/byprot/utils/protein/tokenize_pdb.py"),
            Path("../../src/byprot/utils/protein/tokenize_pdb.py"),
        ]
    )

    try:
        byprot_module = importlib.import_module("byprot")
        byprot_root = Path(byprot_module.__file__).resolve().parent
        candidates.append(byprot_root / "utils/protein/tokenize_pdb.py")
    except Exception:
        pass

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def load_ha(
    ha_path: Path,
    structure_dir: Optional[Path],
    struct_seq_fasta: Optional[Path],
    # 文件扩展名
    structure_exts: Tuple[str, ...] = (".pt", ".npy", ".npz", ".pdb", ".cif", ".fa", ".fasta", ".txt"),
    recursive_structure_search: bool = True,
) -> Dict[int, VirusRecord]:
    ha = pd.read_csv(ha_path)
    required = {"index", "name", "location", "id", "year", "seq"}
    missing = required - set(ha.columns)
    if missing:
        raise ValueError(f"HA file is missing columns: {sorted(missing)}")

    structure_index = build_structure_file_index(structure_dir, structure_exts, recursive_structure_search)
    struct_seq_index = load_struct_seq_index(struct_seq_fasta)
    records: Dict[int, VirusRecord] = {}
    for row in ha.itertuples(index=False):
        virus_index = int(getattr(row, "index"))
        virus_id = str(getattr(row, "id"))
        name = str(getattr(row, "name"))
        structure_path = find_structure_file(structure_index, name.replace('/', ''))
        struct_seq = find_struct_seq(struct_seq_index, name.replace('/', ''))
        if struct_seq is None or structure_path is None:
            print(name)
            exit()
        records[virus_index] = VirusRecord(
            index=virus_index,
            name=name,
            location=str(getattr(row, "location")),
            virus_id=virus_id,
            year=int(getattr(row, "year")),
            seq=str(getattr(row, "seq")).strip().upper(),
            structure_path=structure_path,
            struct_seq=struct_seq,
        )
    matched_struct_seq = sum(1 for record in records.values() if record.struct_seq)
    matched_structure_file = sum(1 for record in records.values() if record.structure_path is not None)
    print(
        f"[data] HA rows={len(records)} "
        f"matched_structure_files={matched_structure_file} "
        f"matched_struct_seq={matched_struct_seq} "
        f"struct_seq_fasta={struct_seq_fasta}"
    )
    return records

'''
struct_seq.fasta
>virus_id1
A,C,G,...
>virus_id2
records = { "virus_id1": "ACG...", "virus_id2": "..." }
'''
def load_struct_seq_index(struct_seq_fasta: Optional[Path]) -> Dict[str, str]:
    if struct_seq_fasta is None or not struct_seq_fasta.exists():
        print(f"[data] struct_seq_fasta not found: {struct_seq_fasta}")
        return {}

    records: Dict[str, str] = {}
    current_id: Optional[str] = None
    chunks: List[str] = []
    with open(struct_seq_fasta, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks).strip()
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            records[current_id] = "".join(chunks).strip()
    print(f"[data] loaded struct_seq records={len(records)} from {struct_seq_fasta}")
    return records


def find_struct_seq(struct_seq_index: Dict[str, str], name: str) -> Optional[str]:
    if name in struct_seq_index:
        return struct_seq_index[name]
    return None


# 去掉所有非字母数字字符，把字母小写化，生成干净、统一的标识符。
def compact_identifier(name: str) -> str:
    # return "".join(ch.lower() for ch in str(name) if ch.isalnum())
    return "".join(ch.lower() for ch in str(name) if ch != '/')


'''
structure_dir/...
 ├─ 1abc.pdb
 ├─ 2xyz.pt
 └─ subdir/3def.fa
↓ build_structure_file_index
index = {
    "1abc": Path(...),
    "1abc.pdb": Path(...),
    "2xyz": Path(...),
    "3def": Path(...),
    ...
}
'''
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
        key = path.stem
        index.setdefault(key, path)
    return index


def find_structure_file(structure_index: Dict[str, Path], name: str) -> Optional[Path]:
    if name in structure_index:
        return structure_index[name]
    return None


def short_sequence_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]


# 按照 HA 氨基酸序列 seq 对病毒记录进行分组；如果多个病毒的序列完全相同，就让它们共享同一个结构文件 structure_path 和同一个结构 token 序列 struct_seq；最后返回一个 DataFrame，用来记录“谁共享了谁的结构”。
def deduplicate_structures_by_sequence(records: Dict[int, VirusRecord]) -> pd.DataFrame:
    #seq_groups: {"AAAA": [病毒A, 病毒B, 病毒D], "BBBB": [病毒C], "CCCC": [病毒E]}
    seq_groups: Dict[str, List[VirusRecord]] = {}
    for record in records.values():
        seq_groups.setdefault(record.seq, []).append(record)
    
    rows = []
    for seq, group in seq_groups.items():
        group = sorted(group, key=lambda item: item.index)
        structure_records = [record for record in group if record.structure_path is not None]
        struct_seq_records = [record for record in group if record.struct_seq is not None]
        shared_path = structure_records[0].structure_path if structure_records else None
        shared_struct_seq = struct_seq_records[0].struct_seq if struct_seq_records else None
        representative_index = structure_records[0].index if structure_records else group[0].index
        representative_id = structure_records[0].virus_id if structure_records else group[0].virus_id

        for record in group:
            original_path = record.structure_path
            record.structure_path = shared_path
            record.struct_seq = shared_struct_seq
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
                    "has_shared_struct_seq": shared_struct_seq is not None,
                    "shared_group_size": len(group),
                }
            )
    return pd.DataFrame(rows).sort_values(["seq_hash", "index"])


def load_hi(
    hi_path: Path,
    virus_records: Dict[int, VirusRecord],
    require_structure: bool,
    require_struct_seq: bool = False,
) -> pd.DataFrame:
    hi = pd.read_csv(hi_path)
    required = {"at_index", "sr_index", "max_year", "min_year", "distance", "class"}
    missing = required - set(hi.columns)
    if missing:
        raise ValueError(f"HI file is missing columns: {sorted(missing)}")

    valid_indices = set(virus_records)
    hi = hi[hi["at_index"].isin(valid_indices) & hi["sr_index"].isin(valid_indices)].copy()
    hi["distance"] = pd.to_numeric(hi["distance"], errors="coerce")
    hi = hi.dropna(subset=["distance"])
    print(f"[data] HI rows after HA-index filtering={len(hi)}")

    if require_structure:
        has_structure = {idx for idx, rec in virus_records.items() if rec.structure_path is not None}
        before = len(hi)
        hi = hi[hi["at_index"].isin(has_structure) & hi["sr_index"].isin(has_structure)].copy()
        print(f"[data] HI rows after structure-file filtering={len(hi)} from {before}")

    if require_struct_seq:
        has_struct_seq = {idx for idx, rec in virus_records.items() if rec.struct_seq}
        before = len(hi)
        hi = hi[hi["at_index"].isin(has_struct_seq) & hi["sr_index"].isin(has_struct_seq)].copy()
        print(
            f"[data] HI rows after struct-seq filtering={len(hi)} from {before}; "
            f"viruses_with_struct_seq={len(has_struct_seq)}/{len(virus_records)}"
        )
    return hi


# 把 HI 表中的两两病毒抗原距离，整理成“每个病毒对应一组邻居病毒及距离”的字典，并按距离从近到远排序。
def build_distance_neighbors(hi: pd.DataFrame) -> Dict[int, List[Tuple[int, float]]]:
    neighbors: Dict[int, Dict[int, float]] = {}
    for row in hi.itertuples(index=False):
        a = int(row.at_index)
        b = int(row.sr_index)
        d = float(row.distance)
        neighbors.setdefault(a, {})
        neighbors.setdefault(b, {})
        # 重复的抗原-抗血清对取最小值
        neighbors[a][b] = min(d, neighbors[a].get(b, d))
        neighbors[b][a] = min(d, neighbors[b].get(a, d))

    return {
        anchor: sorted(items.items(), key=lambda item: item[1])
        for anchor, items in neighbors.items()
        if len(items) >= 2
    }

def pair_key(a: int, b: int) -> Tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def build_pair_distance_dict(hi: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    pair_dist: Dict[Tuple[int, int], float] = {}

    for row in hi.itertuples(index=False):
        a = int(row.at_index)
        b = int(row.sr_index)
        d = float(row.distance)

        if a == b:
            continue

        key = pair_key(a, b)

        # 如果同一对病毒有多条记录，保守地取最小距离
        pair_dist[key] = min(d, pair_dist.get(key, d))

    return pair_dist

def is_positive_pair(
    pair_dist: Dict[Tuple[int, int], float],
    a: int,
    b: int,
    pos_threshold: float = 2.0,
) -> bool:
    d = pair_dist.get(pair_key(a, b))
    return d is not None and d < pos_threshold


# 根据 HI 抗原距离，为每个病毒构造三元组：anchor / positive / negative。
def sample_hi_triplets(
    hi: pd.DataFrame,
    distance_threshold: float,
    distance_scale: float,
    samples_per_anchor: int,
    seed: int,
    mode: str,
) -> List[TripletRecord]:
    # 随机数生成器
    rng = np.random.default_rng(seed)
    neighbors = build_distance_neighbors(hi)

    # 新增：构建任意两个病毒之间的距离查询表
    pair_dist = build_pair_distance_dict(hi)

    triplets: List[TripletRecord] = []
    pos_threshold = 2.0
    neg_threshold = 2.0

    for anchor, candidates in neighbors.items():

        # 1. 去掉 anchor 自己
        candidates = [
            (idx, dist)
            for idx, dist in candidates
            if idx != anchor
        ]

        # 2. 初始 positive 候选：A-B 距离小于 2
        raw_close_candidates = [
            (idx, dist)
            for idx, dist in candidates
            if dist < pos_threshold
        ]

        # 3. 初始 negative 候选：A-D 距离大于 2
        raw_far_candidates = [
            (idx, dist)
            for idx, dist in candidates
            if dist > neg_threshold
        ]

        if not raw_close_candidates or not raw_far_candidates:
            continue

        # 4. 新增条件一：
        # 如果 A-B 和 A-C 都是 positive，那么 B-C 也必须相似
        close_candidates = []
        for pos_index, pos_distance in raw_close_candidates:
            valid_positive = True

            for other_pos_index, _ in raw_close_candidates:
                if other_pos_index == pos_index:
                    continue

                # 要求 positive 集合内部两两相似
                if not is_positive_pair(
                    pair_dist,
                    pos_index,
                    other_pos_index,
                    pos_threshold=pos_threshold,
                ):
                    valid_positive = False
                    break

            if valid_positive:
                close_candidates.append((pos_index, pos_distance))

        if not close_candidates:
            continue

        # 5. 新增条件二：
        # 如果 A-D 是 negative，那么 D 不能和 A 的任何 positive 相似
        far_candidates = []
        for neg_index, neg_distance in raw_far_candidates:
            valid_negative = True

            for pos_index, _ in close_candidates:
                # 如果 positive 和 negative 之间也相似，则这个 negative 不可靠
                if is_positive_pair(
                    pair_dist,
                    pos_index,
                    neg_index,
                    pos_threshold=pos_threshold,
                ):
                    valid_negative = False
                    break

            if valid_negative:
                far_candidates.append((neg_index, neg_distance))

        if not far_candidates:
            continue

        if mode == "all":
            for pos_index, pos_distance in close_candidates:
                for neg_index, neg_distance in far_candidates:

                    # 6. anchor、positive、negative 三者必须不同
                    if pos_index == anchor or neg_index == anchor:
                        continue
                    if pos_index == neg_index:
                        continue

                    # 7. negative 和 positive 与 anchor 的距离差必须足够大
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

            pos_index, pos_distance = close_candidates[
                int(rng.integers(0, len(close_candidates)))
            ]

            neg_index, neg_distance = far_candidates[
                int(rng.integers(0, len(far_candidates)))
            ]

            if pos_index == anchor or neg_index == anchor:
                continue
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


def decode_sequence(seq_ids: torch.Tensor) -> str:
    return "".join(ID_TO_AA.get(int(idx), "X") for idx in seq_ids if int(idx) != PAD_ID)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


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
            "aa_seq": record.seq,
            "struct_seq": record.struct_seq or "",
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
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


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

    def forward(
        self,
        seq_ids: torch.Tensor,
        structure: torch.Tensor,
        aa_seq: Optional[List[str]] = None,
        struct_seq: Optional[List[str]] = None,
        index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        mismatch_strategy: str = "gap",
        max_mismatch_ratio: float = 0.1,
        aa_gap_token: str = "X",
        struct_missing_token: str = "auto",
    ):
        super().__init__()
        self.mismatch_strategy = mismatch_strategy
        self.max_mismatch_ratio = max_mismatch_ratio
        self.aa_gap_token = aa_gap_token
        self.struct_missing_token = struct_missing_token
        self._mismatch_warnings = 0
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

    def forward(
        self,
        seq_ids: Optional[torch.Tensor] = None,
        structure: Optional[torch.Tensor] = None,
        aa_seq: Optional[List[str]] = None,
        struct_seq: Optional[List[str]] = None,
        index: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is None:
            input_ids = self.build_input_ids(seq_ids=seq_ids, aa_seq=aa_seq, struct_seq=struct_seq)
        self.validate_input_ids(input_ids)

        output = self.model(input_ids=input_ids)
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            if "last_hidden_state" in output:
                return self.pool_multimodal_hidden(output["last_hidden_state"], input_ids)
            for key in ("pooler_output", "embedding", "embeddings"):
                if key in output:
                    return output[key]
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return self.pool_multimodal_hidden(output.last_hidden_state, input_ids)
        raise TypeError("Unsupported dplm-2 output type. Please adapt DPLM2Backbone.forward().")

    def build_input_ids(
        self,
        seq_ids: Optional[torch.Tensor],
        aa_seq: Optional[List[str]],
        struct_seq: Optional[List[str]],
    ) -> torch.Tensor:
        if aa_seq is None:
            if seq_ids is None:
                raise ValueError("DPLM2Backbone needs aa_seq or seq_ids.")
            aa_seq = [decode_sequence(row) for row in seq_ids.detach().cpu()]
        elif isinstance(aa_seq, str):
            aa_seq = [aa_seq]

        if struct_seq is None:
            raise ValueError(
                "DPLM2Backbone needs struct_seq. Provide --struct-seq-fasta or structure files containing struct_seq."
            )
        elif isinstance(struct_seq, str):
            struct_seq = [struct_seq]

        aa_inputs = []
        struct_inputs = []
        tokenizer = self.get_tokenizer()
        for aa, struct in zip(aa_seq, struct_seq):
            if not struct:
                raise ValueError(f"Missing struct_seq for aa sequence with length {len(aa)}.")
            struct_tokens = struct.split(",") if "," in struct else list(struct)
            if len(aa) != len(struct_tokens):
                aa, struct_tokens = self.handle_length_mismatch(aa, struct_tokens, tokenizer)
            else:
                aa = self.replace_aa_gaps(aa)
            struct_joined = "".join(token.strip() for token in struct_tokens)
            aa_inputs.append(tokenizer.aa_cls_token + aa + tokenizer.aa_eos_token)
            struct_inputs.append(
                tokenizer.struct_cls_token
                + struct_joined
                + tokenizer.struct_eos_token
            )

        aa_batch = tokenizer.batch_encode_plus(
            aa_inputs,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        struct_batch = tokenizer.batch_encode_plus(
            struct_inputs,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        aa_ids = aa_batch["input_ids"].to(device)
        struct_ids = struct_batch["input_ids"].to(device)
        return torch.cat([struct_ids, aa_ids], dim=1)

    def validate_input_ids(self, input_ids: torch.Tensor) -> None:
        embedding = self.get_input_embedding()
        if embedding is None:
            return
        vocab_size = embedding.num_embeddings
        min_id = int(input_ids.min().detach().cpu())
        max_id = int(input_ids.max().detach().cpu())
        if min_id < 0 or max_id >= vocab_size:
            bad = input_ids[(input_ids < 0) | (input_ids >= vocab_size)]
            examples = bad[:20].detach().cpu().tolist()
            raise ValueError(
                f"DPLM2 input_ids out of embedding range: min={min_id}, max={max_id}, "
                f"vocab_size={vocab_size}, bad_examples={examples}. "
                "This usually means the structure-missing token is not in the model vocabulary. "
                "Try --struct-missing-token '<pad>' or another valid tokenizer pad token."
            )

    def get_input_embedding(self) -> Optional[nn.Embedding]:
        candidates = []
        model = self.model
        candidates.append(model)
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            candidates.append(model.base_model.model)
        for obj in candidates:
            if hasattr(obj, "get_input_embeddings"):
                emb = obj.get_input_embeddings()
                if emb is not None:
                    return emb
            try:
                return obj.net.esm.embeddings.word_embeddings
            except AttributeError:
                pass
        return None

    def handle_length_mismatch(self, aa: str, struct_tokens: List[str], tokenizer) -> Tuple[str, List[str]]:
        aa_len = len(aa)
        struct_len = len(struct_tokens)
        mismatch = abs(aa_len - struct_len)
        mismatch_ratio = mismatch / max(aa_len, struct_len, 1)

        if self.mismatch_strategy == "gap":
            gap_count = aa.count("-")
            if struct_len == aa_len - gap_count:
                missing_token = self.resolve_struct_missing_token(tokenizer)
                aligned_struct_tokens = []
                struct_pos = 0
                for aa_char in aa:
                    if aa_char == "-":
                        aligned_struct_tokens.append(missing_token)
                    else:
                        aligned_struct_tokens.append(struct_tokens[struct_pos])
                        struct_pos += 1
                aa_for_model = self.replace_aa_gaps(aa)
                if self._mismatch_warnings < 20:
                    print(
                        f"[dplm2] gap-aware alignment aa={aa_len}, struct={struct_len}, "
                        f"aa_gaps={gap_count}; inserted {gap_count} structure-missing tokens "
                        f"({missing_token})"
                    )
                    self._mismatch_warnings += 1
                return aa_for_model, aligned_struct_tokens

            raise ValueError(
                f"Length mismatch cannot be explained by '-' gaps: aa={aa_len}, "
                f"struct={struct_len}, aa_gaps={gap_count}. "
                "Please check whether the PDB sequence matches the HA sequence."
            )

        if self.mismatch_strategy == "error" or mismatch_ratio > self.max_mismatch_ratio:
            raise ValueError(
                f"Length mismatch between aa_seq and struct_seq: aa={aa_len}, struct={struct_len}. "
                f"mismatch_ratio={mismatch_ratio:.4f}, strategy={self.mismatch_strategy}"
            )
        if self.mismatch_strategy != "trim":
            raise ValueError(f"Unsupported mismatch strategy: {self.mismatch_strategy}")

        keep_len = min(aa_len, struct_len)
        if self._mismatch_warnings < 20:
            print(
                f"[dplm2] length mismatch aa={aa_len}, struct={struct_len}; "
                f"trim both to {keep_len}"
            )
            self._mismatch_warnings += 1
        return aa[:keep_len], struct_tokens[:keep_len]

    def replace_aa_gaps(self, aa: str) -> str:
        return aa.replace("-", self.aa_gap_token)

    def resolve_struct_missing_token(self, tokenizer) -> str:
        if self.struct_missing_token != "auto":
            return self.struct_missing_token
        for attr in (
            "struct_pad_token",
            "pad_token",
            "struct_unk_token",
            "unk_token",
            "struct_mask_token",
            "mask_token",
        ):
            token = getattr(tokenizer, attr, None)
            if token:
                return token
        raise ValueError(
            "Cannot infer a structure-missing token from tokenizer. "
            "Please set --struct-missing-token explicitly."
        )

    def get_tokenizer(self):
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
            inner = self.model.base_model.model
            if hasattr(inner, "tokenizer"):
                return inner.tokenizer
        raise AttributeError("Cannot find tokenizer on dplm-2 model.")

    def pool_multimodal_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        pad_id = getattr(self.model, "pad_id", None)
        if pad_id is None and hasattr(self.model, "base_model"):
            pad_id = getattr(self.model.base_model.model, "pad_id", None)
        if pad_id is None:
            pad_id = 1

        half = hidden.shape[1] // 2
        struct_hidden = hidden[:, :half, :]
        aa_hidden = hidden[:, half:, :]
        struct_mask = input_ids[:, :half].ne(pad_id)
        aa_mask = input_ids[:, half:].ne(pad_id)
        struct_feature = masked_mean(struct_hidden[:, 1:-1, :], struct_mask[:, 1:-1])
        aa_feature = masked_mean(aa_hidden[:, 1:-1, :], aa_mask[:, 1:-1])
        return torch.cat([aa_feature, struct_feature], dim=-1)


def maybe_apply_lora(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("peft is not installed. Install peft or run without --use-lora.")

    lora_root = model.model if isinstance(model, DPLM2Backbone) else model
    target_modules = resolve_lora_targets(lora_root, args.lora_targets)
    print(f"[lora] target_modules={target_modules}")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    lora_model = get_peft_model(lora_root, config)
    if isinstance(model, DPLM2Backbone):
        model.model = lora_model
        return model
    return lora_model


def resolve_lora_targets(model: nn.Module, target_spec: str) -> List[str]:
    requested = [item.strip() for item in target_spec.split(",") if item.strip()]
    linear_suffixes = sorted(
        {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name
        }
    )
    if not linear_suffixes:
        raise ValueError("No nn.Linear modules were found for LoRA injection.")

    if requested and requested != ["auto"]:
        missing = [name for name in requested if name not in linear_suffixes]
        if not missing:
            return requested
        preview = ", ".join(linear_suffixes[:80])
        raise ValueError(
            f"LoRA target modules not found: {missing}. "
            f"Available nn.Linear suffixes include: {preview}"
        )

    priority_keywords = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "out_proj",
        "qkv",
        "fc1",
        "fc2",
        "fc",
        "wq",
        "wk",
        "wv",
        "wo",
    )
    targets = [
        suffix
        for suffix in linear_suffixes
        if any(keyword in suffix.lower() for keyword in priority_keywords)
    ]

    if not targets:
        targets = linear_suffixes

    return targets


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
    return DPLM2Backbone(
        args.dplm_module,
        args.dplm_class,
        args.dplm_model,
        args.dplm_struct_tokenizer_dir,
        mismatch_strategy=args.length_mismatch_strategy,
        max_mismatch_ratio=args.max_mismatch_ratio,
        aa_gap_token=args.aa_gap_token,
        struct_missing_token=args.struct_missing_token,
    )


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
    return model(**features)


@torch.no_grad()
def evaluate_stage1_dpo_loss(
    policy: nn.Module,
    reference: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    beta: float,
    lambda_triplet: float,
) -> float:
    if loader is None:
        return float("nan")
    policy.eval()
    losses: List[float] = []
    for batch in loader:
        anchor = move_feature_batch(batch["anchor"], device)
        positive = move_feature_batch(batch["positive"], device)
        negative = move_feature_batch(batch["negative"], device)
        margin = batch["margin"].to(device)

        p_a = encode_batch(policy, anchor)
        p_p = encode_batch(policy, positive)
        p_n = encode_batch(policy, negative)
        r_a = encode_batch(reference, anchor)
        r_p = encode_batch(reference, positive)
        r_n = encode_batch(reference, negative)

        loss_dpo = dpo_triplet_loss(p_a, p_p, p_n, r_a, r_p, r_n, beta)
        loss_triplet = margin_triplet_loss(p_a, p_p, p_n, margin)
        loss = loss_dpo + lambda_triplet * loss_triplet
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_stage1_dpo(
    args: argparse.Namespace,
    triplets: List[TripletRecord],
    feature_store: VirusFeatureStore,
    output_dir: Path,
    device: torch.device,
    val_triplets: Optional[List[TripletRecord]] = None,
) -> nn.Module:
    policy = build_backbone(args)
    reference = build_backbone(args).to(device)
    reference.load_state_dict(policy.state_dict(), strict=False)
    policy = maybe_apply_lora(policy, args).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    dataset = TripletDataset(triplets, feature_store)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if val_triplets:
        val_dataset = TripletDataset(val_triplets, feature_store)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epoch_rows = []

    policy.train()
    for epoch in range(1, args.stage1_epochs + 1):
        losses = []
        for step, batch in enumerate(tqdm(loader), start=1):
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
            
            # 关键：loss 除以累积步数，避免梯度被放大
            loss_for_backward = loss / args.gradient_accumulation_steps
            loss_for_backward.backward()
            # 记录原始 loss，不记录除过的 loss
            losses.append(float(loss.detach().cpu()))


            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            # optimizer.step()
            # losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = evaluate_stage1_dpo_loss(policy, reference, val_loader, device, args.beta, args.lambda_triplet) if val_loader is not None else float("nan")
        monitor_loss = val_loss if val_loader is not None else train_loss
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "monitor_loss": monitor_loss})
        if monitor_loss + args.early_stopping_min_delta < best_val_loss:
            best_val_loss = monitor_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        else:
            bad_epochs += 1

        if val_loader is not None:
            print(f"[stage1] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_epoch={best_epoch}")
        else:
            print(f"[stage1] epoch={epoch} loss={train_loss:.6f}")

        if val_loader is not None and args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            print(f"[stage1] early stopping at epoch={epoch}; best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
            break

    if best_state is not None:
        policy.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    pd.DataFrame(epoch_rows).to_csv(output_dir / "stage1_epoch_log.csv", index=False)

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
            nn.LazyLinear(max(embedding_dim // 2, 2)),
            nn.GELU(),
            nn.LayerNorm(max(embedding_dim // 2, 2)),
            nn.Linear(max(embedding_dim // 2, 2), 2),
        )
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        seq_ids: torch.Tensor,
        structure: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        aa_seq: Optional[List[str]] = None,
        struct_seq: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if seq_ids.dim() == 1:
            seq_ids = seq_ids.unsqueeze(0)
        if structure.dim() == 1:
            structure = structure.unsqueeze(0)
        embedding = self.backbone(seq_ids=seq_ids, structure=structure, aa_seq=aa_seq, struct_seq=struct_seq, index=index)
        return self.projector(embedding)


def variance_regularizer(points: torch.Tensor, min_variance: float) -> torch.Tensor:
    variance = points.var(dim=0, unbiased=False).mean()
    return F.relu(torch.tensor(min_variance, device=points.device) - variance)


@torch.no_grad()
def evaluate_stage2_pair_loss(
    model: AntigenMapModel,
    pair_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validation loss on labeled pairs in normalized antigenic-distance scale."""
    if pair_loader is None:
        return float("nan")
    model.eval()
    losses: List[float] = []
    for batch in pair_loader:
        left = move_feature_batch(batch["left"], device)
        right = move_feature_batch(batch["right"], device)
        target_distance = batch["distance"].to(device)
        y_left = model(**left)
        y_right = model(**right)
        map_distance = torch.linalg.vector_norm(y_left - y_right, dim=-1)
        loss = F.mse_loss(map_distance, target_distance)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_stage2_map(
    args: argparse.Namespace,
    backbone: nn.Module,
    hi: pd.DataFrame,
    seq_triplets: List[TripletRecord],
    feature_store: VirusFeatureStore,
    virus_records: Dict[int, VirusRecord],
    output_dir: Path,
    device: torch.device,
    val_hi: Optional[pd.DataFrame] = None,
) -> AntigenMapModel:
    distance_min = float(hi["distance"].min())
    distance_max = float(hi["distance"].max())
    pair_dataset = PairDataset(hi, feature_store, distance_min, distance_max)
    pair_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if val_hi is not None and len(val_hi) > 0:
        val_dataset = PairDataset(val_hi, feature_store, distance_min, distance_max)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    seq_dataset = TripletDataset(seq_triplets, feature_store) if seq_triplets else None
    seq_loader = (
        iter(DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers))
        if seq_dataset is not None
        else None
    )

    model = AntigenMapModel(backbone, args.embedding_dim, args.train_backbone_stage2).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr_stage2, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epoch_rows = []

    for epoch in range(1, args.stage2_epochs + 1):
        losses = []
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(pair_loader), start=1):
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
            loss_for_backward = loss / args.gradient_accumulation_steps
            loss_for_backward.backward()
            losses.append(float(loss.detach().cpu()))

            if step % args.gradient_accumulation_steps == 0 or step == len(pair_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = evaluate_stage2_pair_loss(model, val_loader, device) if val_loader is not None else float("nan")
        monitor_loss = val_loss if val_loader is not None else train_loss
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "monitor_loss": monitor_loss})

        if monitor_loss + args.early_stopping_min_delta < best_val_loss:
            best_val_loss = monitor_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1

        if val_loader is not None:
            print(f"[stage2] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_epoch={best_epoch}")
        else:
            print(f"[stage2] epoch={epoch} loss={train_loss:.6f}")

        if val_loader is not None and args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            print(f"[stage2] early stopping at epoch={epoch}; best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    pd.DataFrame(epoch_rows).to_csv(output_dir / "stage2_epoch_log.csv", index=False)

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


def save_virus_input_mapping(records: Dict[int, VirusRecord], path: Path) -> None:
    rows = []
    for record in sorted(records.values(), key=lambda item: item.index):
        rows.append(
            {
                "index": record.index,
                "id": record.virus_id,
                "name": record.name,
                "compact_name": compact_identifier(record.name),
                "seq_length": len(record.seq),
                "structure_path": str(record.structure_path) if record.structure_path else "",
                "has_struct_seq": bool(record.struct_seq),
                "struct_seq_length": len(record.struct_seq.split(",")) if record.struct_seq and "," in record.struct_seq else len(record.struct_seq or ""),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPLM-2 HI-distance DPO and antigen-map training pipeline.")
    parser.add_argument("--ha", type=Path, default=DEFAULT_HA_PATH, help="HA csv: index,name,location,id,year,seq")
    parser.add_argument("--hi", type=Path, default=DEFAULT_HI_PATH, help="HI csv: at_index,sr_index,max_year,min_year,distance,class")
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--struct-seq-fasta", type=Path, default=DEFAULT_STRUCT_SEQ_FASTA)
    parser.add_argument("--auto-tokenize-structures", dest="auto_tokenize_structures", action="store_true")
    parser.add_argument("--no-auto-tokenize-structures", dest="auto_tokenize_structures", action="store_false")
    parser.set_defaults(auto_tokenize_structures=True)
    parser.add_argument("--pdb-tokenizer-script", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--subtype", choices=["auto", "H3", "H5", "unknown"], default="auto", help="Subtype-specific output namespace. Use auto to infer from input paths.")
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--no-recursive-structure-search", action="store_true", default=False)
    parser.add_argument("--no-dedupe-structure-by-seq", action="store_true", default=False)

    parser.add_argument("--encoder", choices=["simple", "dplm2"], default="dplm2")
    parser.add_argument("--dplm-module", default="byprot.models.dplm2")
    parser.add_argument("--dplm-class", default="MultimodalDiffusionProteinLanguageModel")
    parser.add_argument("--dplm-model", type=Path, default=DEFAULT_DPLM_MODEL)
    parser.add_argument("--dplm-struct-tokenizer-dir", type=Path, default=DEFAULT_DPLM_STRUCT_TOKENIZER_DIR)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--structure-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--length-mismatch-strategy", choices=["gap", "trim", "error"], default="gap")
    parser.add_argument("--aa-gap-token", default="X")
    parser.add_argument("--struct-missing-token", default="auto")
    parser.add_argument("--max-mismatch-ratio", type=float, default=0.1)

    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--use-lora", dest="use_lora", action="store_true")
    lora_group.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="auto")

    parser.add_argument("--distance-threshold", type=float, default=1.0)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--hi-triplet-mode", choices=["sample", "all"], default="all")
    parser.add_argument("--hi-triplets-per-anchor", type=int, default=4)
    parser.add_argument("--seq-threshold", type=float, default=0.05)
    parser.add_argument("--seq-scale", type=float, default=1.0)
    parser.add_argument("--seq-triplets-per-anchor", type=int, default=2)

    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Stop stage2 when validation loss does not improve; <=0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=3)
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=43)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

