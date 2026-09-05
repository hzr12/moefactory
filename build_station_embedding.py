"""基于站间距矩阵，用 MDS 将车站距离降维为连续 embedding。

两种输入：
  A. 单线/少量线：datasets/station_mileage.csv（fetch_jprailfan_mileage.py 单线模式产出）
     站间距 = |累计km_i - 累计km_j|  →  对称距离矩阵  →  MDS
  B. 全路网：datasets/network.json（fetch_jprailfan_mileage.py --all-lines 产出）
     把每条线路的站点序列还原成相邻站间距(图边)，在全网图上求“最短路距离”矩阵，
     再 MDS。这样枢纽站(跨多条线)也能正确定位，且距离即铁路网可达距离。

输出：model/station_embedding.pkl  ——  {站名: np.ndarray(K,)}，按站名索引

原理：距离是静态地理属性，不经过延误标签 → 验证集划分不受影响（零泄漏）。

用法：
  python build_station_embedding.py --k 2                       # 单线 csv（默认）
  python build_station_embedding.py --from-json datasets/network.json --k 16
"""
import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import MDS


def _build_from_csv(mileage_csv, k, random_state):
    df = pd.read_csv(mileage_csv)
    names = df["车站名"].astype(str).tolist()
    km = df["累计km"].astype(float).values
    D = np.abs(km[:, None] - km[None, :]).astype(float)
    emb = _mds(D, k, random_state)
    return names, emb


def _build_from_network(network_json, k, random_state):
    """由全网线路序列构建最短路距离矩阵并 MDS。需要 scipy。"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    with open(network_json, encoding="utf-8") as f:
        lines = json.load(f).get("lines", {})
    print(f"载入 {len(lines)} 条线路的站点序列")

    stations = sorted({s for seq in lines.values() for s, _ in seq})
    idx = {s: i for i, s in enumerate(stations)}
    n = len(stations)
    print(f"去重后 {n} 个站")

    # 边：每条线路相邻站 -> 权重 = |累计km 差|（即站间距）；多线重合取最小
    edge_w = {}
    for seq in lines.values():
        for (s1, c1), (s2, c2) in zip(seq, seq[1:]):
            w = abs(c2 - c1)
            if w <= 0:
                continue
            a, b = idx[s1], idx[s2]
            key = (a, b) if a < b else (b, a)
            if key not in edge_w or w < edge_w[key]:
                edge_w[key] = w

    rows = []
    cols = []
    weights = []
    for (a, b), w in edge_w.items():
        rows += [a, b]
        cols += [b, a]
        weights += [w, w]
    G = csr_matrix((weights, (rows, cols)), shape=(n, n))
    print(f"构图完成：{len(edge_w)} 条无向边")

    # 最短路距离矩阵（全成对）
    dist = shortest_path(G, method="D", directed=False)  # (n, n) float64
    inf_mask = ~np.isfinite(dist)
    n_inf = int(inf_mask.sum())
    if n_inf:
        finite_max = np.nanmax(dist[~inf_mask]) if (~inf_mask).any() else 0.0
        # 不连通对：用 2×最大有限距离封顶，避免 MDS 崩溃（仍可后续按连通分量拆分）
        dist[inf_mask] = finite_max * 2.0
        print(f"  [注意] {n_inf} 对站不连通，已封顶为 {finite_max * 2:.0f} km")
    D = dist.astype(float)
    emb = _mds(D, k, random_state)
    return stations, emb


def _mds(D, k, random_state):
    # dissimilarity='precomputed' 在新版 sklearn 会发 FutureWarning，此处抑制
    warnings.filterwarnings("ignore", category=FutureWarning)
    mds = MDS(n_components=k, dissimilarity="precomputed", init="random",
              random_state=random_state)
    return mds.fit_transform(D)


def main():
    ap = argparse.ArgumentParser(description="由站间距矩阵 MDS 生成 station_embedding")
    ap.add_argument("--mileage", default="datasets/station_mileage.csv")
    ap.add_argument("--from-json", default=None,
                    help="全路网模式：传入 datasets/network.json")
    ap.add_argument("--out", default="model/station_embedding.pkl")
    ap.add_argument("--k", type=int, default=2, help="嵌入维度")
    ap.add_argument("--random-state", type=int, default=3461)
    args = ap.parse_args()

    if args.from_json:
        if not os.path.exists(args.from_json):
            raise SystemExit(f"找不到 {args.from_json}，请先运行 fetch_jprailfan_mileage.py --all-lines")
        names, emb = _build_from_network(args.from_json, args.k, args.random_state)
    else:
        if not os.path.exists(args.mileage):
            raise SystemExit(f"找不到 {args.mileage}，请先运行 fetch_jprailfan_mileage.py")
        names, emb = _build_from_csv(args.mileage, args.k, args.random_state)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    emb_dict = {name: emb[i] for i, name in enumerate(names)}
    with open(args.out, "wb") as f:
        pickle.dump(emb_dict, f)
    print(f"已生成 {args.out}：{len(names)} 站 × {args.k} 维")
    for name in names[:6]:
        print(f"  {name}: {np.round(emb_dict[name], 3)}")


if __name__ == "__main__":
    main()
