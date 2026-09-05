"""可视化 station embedding:全路网最短路距离 → 2D MDS 地图。

- 灰点:车站(2D MDS 坐标,形态近似真实铁路网)
- 蓝线:主要干线走向;红线:京广高速线(训练线)
- 红星+站名:训练集 44 站;蓝点+站名:主要枢纽
- 朝向:用枢纽城市先验坐标做 Kabsch 相似变换(旋转/翻转/缩放),
  使地图大致符合"上北下南"的地理直觉
- 孤立小分量(几条支线/轮渡线)不参与 2D 拟合,避免封顶距离压缩主图

输出:outputs/station_embedding_map.png
"""
import json
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")

NET = "datasets/network.json"
OUT = "outputs/station_embedding_map.png"

TRUNK = ["京广高速线", "京广线", "京沪高速线", "京沪线", "沪昆高速线", "沪昆线",
         "陇海线", "京九线", "京哈高速线京沈段", "京哈线", "京哈高速线沈哈段",
         "兰新线", "兰新客专线", "青藏线", "杭深线", "宝成线", "成昆线成攀段",
         "滨洲线", "贵广客专线", "徐兰高速线"]

HUBS = ["北京西", "上海虹桥", "广州南", "成都东", "西安北", "武汉", "郑州东",
        "长沙南", "哈尔滨西", "乌鲁木齐", "昆明南", "兰州西", "沈阳北", "杭州东",
        "重庆北", "贵阳北"]

# 枢纽地理先验(x 向东, y 向北, 任意比例)——仅用于定向,不参与拟合精度
ANCHOR_GEO = {
    "乌鲁木齐": (-1900, 300), "拉萨": (-1500, -1300), "昆明南": (200, -1700),
    "贵阳北": (450, -1250), "广州南": (900, -1500), "长沙南": (800, -900),
    "武汉": (850, -350), "郑州东": (800, 250), "北京西": (700, 950),
    "哈尔滨西": (1900, 1500), "沈阳北": (1550, 1150), "上海虹桥": (1650, -300),
    "杭州东": (1450, -450), "西安北": (500, 100), "兰州西": (-100, 50),
    "成都东": (-450, -450), "重庆北": (350, -700),
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 1. 构图 + 最短路距离 ----
d = json.load(open(NET, encoding="utf-8"))["lines"]
stations = sorted({s for seq in d.values() for s, _ in seq})
idx = {s: i for i, s in enumerate(stations)}
n = len(stations)

edge_w = {}
for seq in d.values():
    for (s1, c1), (s2, c2) in zip(seq, seq[1:]):
        w = abs(c2 - c1)
        if w <= 0:
            continue
        a, b = idx[s1], idx[s2]
        key = (a, b) if a < b else (b, a)
        if key not in edge_w or w < edge_w[key]:
            edge_w[key] = w

rows, cols, ws = [], [], []
for (a, b), w in edge_w.items():
    rows += [a, b]
    cols += [b, a]
    ws += [w, w]
G = csr_matrix((ws, (rows, cols)), shape=(n, n))
D = shortest_path(G, method="D", directed=False)

# ---- 2. 只在最大连通分量上做 2D MDS ----
ncomp, labels = connected_components(G, directed=False)
sizes = np.bincount(labels)
keep = np.where(labels == sizes.argmax())[0]
dropped = n - len(keep)
if dropped:
    print(f"最大连通分量 {len(keep)} 站(丢弃 {dropped} 个孤立小站,不影响主图)")
Dsub = D[np.ix_(keep, keep)]
names_sub = [stations[i] for i in keep]

mds = MDS(n_components=2, dissimilarity="precomputed", init="random",
          n_init=1, random_state=3461)
XY = mds.fit_transform(Dsub)
print(f"2D MDS 完成, stress={mds.stress_:.3e}")
pos = {s: XY[i] for i, s in enumerate(names_sub)}

# ---- 3. Kabsch 相似变换对齐地理朝向 ----
ax_ = [pos[s] for s in ANCHOR_GEO if s in pos]
ay_ = [np.array(ANCHOR_GEO[s], float) for s in ANCHOR_GEO if s in pos]
X = np.array(ax_)
Y = np.array(ay_)
Xm, Ym = X.mean(0), Y.mean(0)
Xc, Yc = X - Xm, Y - Ym
U, S, Vt = np.linalg.svd(Xc.T @ Yc)
R = Vt.T @ U.T                       # 允许翻转(MDS 镜像不确定)
s_scale = np.sum((Xc @ R.T) * Yc) / np.sum(Xc ** 2)
t = Ym - s_scale * (R @ Xm)
for k in pos:
    pos[k] = s_scale * (R @ pos[k]) + t
XYa = np.array([pos[s] for s in names_sub])
print(f"Kabsch 对齐: scale={s_scale:.2f}, det(R)={np.linalg.det(R):+.2f}")

# ---- 4. 训练集车站 ----
import data_preprocess as dp
df = dp.load_train_data("./datasets/train")
train_st = sorted(set(df["车站名"].dropna().astype(str).str.strip()))

# ---- 5. 画图 ----
fig, ax = plt.subplots(figsize=(13, 12), dpi=150)

ax.scatter(XYa[:, 0], XYa[:, 1], s=6, c="#9aa5b1", alpha=0.35, linewidths=0, zorder=1)

for ln in TRUNK:
    if ln not in d:
        continue
    pts = np.array([pos[s] for s, _ in d[ln] if s in pos])
    if len(pts) >= 2:
        ax.plot(pts[:, 0], pts[:, 1], c="#5b8db8", lw=1.0, alpha=0.55, zorder=2)

if "京广高速线" in d:
    pts = np.array([pos[s] for s, _ in d["京广高速线"] if s in pos])
    ax.plot(pts[:, 0], pts[:, 1], c="#d64541", lw=2.2, alpha=0.9, zorder=3)

tx = np.array([pos[s] for s in train_st if s in pos])
ax.scatter(tx[:, 0], tx[:, 1], s=70, c="#d64541", marker="*", zorder=4,
           edgecolors="white", linewidths=0.5, label=f"训练集车站 G339 沿线 (n={len(tx)})")

texts = []
for s in train_st:
    if s not in pos:
        continue
    texts.append(ax.text(pos[s][0], pos[s][1], s, fontsize=7.5, color="#8b1a1a", zorder=6))
for s in HUBS:
    if s not in pos:
        continue
    ax.scatter([pos[s][0]], [pos[s][1]], s=18, c="#1f4e79", zorder=5)
    texts.append(ax.text(pos[s][0], pos[s][1], s, fontsize=9, color="#1f4e79",
                         fontweight="bold", zorder=6))

try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax,
                expand=(1.15, 1.3),
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5))
    print("标签防重叠: adjustText")
except ImportError:
    print("标签防重叠: 未安装 adjustText,使用原始位置(建议 pip install adjustText)")

ax.set_title(f"全国铁路网车站 embedding(最短路距离 MDS → 2D,Kabsch 定向)\n"
             f"{n} 站 / {len(d)} 条线路 / {len(edge_w)} 条区间边  |  红线=京广高速线",
             fontsize=13)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_alpha(0.2)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"已保存 {OUT}")
