import numpy as np
import pandas as pd

# -----------------------------
# 1. 准备一个极小 toy 数据
# -----------------------------
virus2seq = {
    "A/virus_2018": "AAABBBCCC",
    "B/virus_2018": "AAABBBCCA",  # 距 A 很近：1 位不同
    "C/virus_2019": "AAABBBAAA",  # 中等差异
    "D/virus_2020": "CCCCCCCCC",  # 距 A 很远
    "E/virus_2021": "GGGGGGGGG",  # 距 A 很远
}

virus_names = list(virus2seq.keys())


# -----------------------------
# 2. 计算 pairwise 序列差异矩阵
# -----------------------------
def seq_diff(seq1, seq2):
    return sum(a != b for a, b in zip(seq1, seq2))


n = len(virus_names)
diff_matrix = np.zeros((n, n), dtype=int)

for i, v1 in enumerate(virus_names):
    for j, v2 in enumerate(virus_names):
        diff_matrix[i, j] = seq_diff(virus2seq[v1], virus2seq[v2])

diff_df = pd.DataFrame(diff_matrix, index=virus_names, columns=virus_names)
print("=== 序列差异矩阵 ===")
print(diff_df)


# -----------------------------
# 3. 基于差异采样 contrastive triplet
# -----------------------------
def sample_triplets(
    anchor_names,
    candidate_pool,
    diff_threshold=3,
    diff_scale=2,
    n_samples_per_anchor=2,
    seed=42,
):
    rng = np.random.default_rng(seed)

    name_to_idx = {name: i for i, name in enumerate(virus_names)}
    triplets = []

    for anchor in anchor_names:
        anchor_idx = name_to_idx[anchor]

        candidates = []
        for cand in candidate_pool:
            if cand == anchor:
                continue

            cand_idx = name_to_idx[cand]
            diff = diff_matrix[anchor_idx, cand_idx]

            if diff > 0:
                candidates.append((cand, diff))

        # 按与 anchor 的差异从小到大排序
        candidates = sorted(candidates, key=lambda x: x[1])

        sampled = 0
        attempts = 0

        while sampled < n_samples_per_anchor and attempts < 20:
            attempts += 1

            # positive：从差异较小的一半里选
            pos_name, pos_diff = candidates[
                rng.integers(0, max(1, len(candidates) // 2))
            ]

            # negative：从差异较大的一半里选
            neg_name, neg_diff = candidates[
                rng.integers(len(candidates) // 2, len(candidates))
            ]

            diff_gap = neg_diff - pos_diff

            # 要求 negative 比 positive 至少远 diff_threshold
            if diff_gap < diff_threshold:
                continue

            margin = diff_scale * diff_gap / diff_threshold

            triplets.append({
                "anchor": anchor,
                "positive": pos_name,
                "negative": neg_name,
                "pos_diff": pos_diff,
                "neg_diff": neg_diff,
                "diff_gap": diff_gap,
                "margin": margin,
            })

            sampled += 1

    return triplets


# -----------------------------
# 4. Demo：采样并展示结果
# -----------------------------
anchors = ["A/virus_2018", "C/virus_2019"]
candidate_pool = virus_names

triplets = sample_triplets(
    anchors,
    candidate_pool,
    diff_threshold=3,
    diff_scale=2,
    n_samples_per_anchor=2,
)

print("\n=== 采样得到的对比学习三元组 ===")
for t in triplets:
    print(
        f"anchor={t['anchor']} | "
        f"positive={t['positive']} diff={t['pos_diff']} | "
        f"negative={t['negative']} diff={t['neg_diff']} | "
        f"gap={t['diff_gap']} | "
        f"margin={t['margin']:.2f}"
    )