"""从 jprailfan.com 抓取铁路线路每站累计里程。

数据来源：黄河铁路网「车站信息查询」系统
  - 所有线路列表页：tools/stat/?key7=所有线路输出到本页
  - 线路详情页（含每站累计里程）：tools/stat/?linename=<线路名>
    详情页“下面显示该线路的里程表”段给出每站的「距起始站里程」(累计 km) 与「相邻站里程」。

两种用法：
  # 单条线路 -> station_mileage.csv（站名, 累计km），供单线 MDS
  python fetch_jprailfan_mileage.py --lines 京广高速线

  # 全路网 -> network.json（保留每条线路的站点顺序，供最短路图嵌入）
  python fetch_jprailfan_mileage.py --all-lines
  python fetch_jprailfan_mileage.py --all-lines --limit 50 --offset 0   # 分批/续跑

说明：
  - 仅依赖标准库（urllib / re / csv / json），不引入新依赖。
  - 网络详情页为 12 列表格，站名在「途经车站」列(首个 ?statinfo= 链接)，里程为 '274km' 格式；
    逐行解析，跳过电报码(-开头)与车站编号(纯数字)等噪声链接。
  - 合规：jprailfan 标注“非官方、仅供参考”，本脚本仅内部抓取算特征、不对外转发原始表。
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "https://jprailfan.com/tools/stat/"
ALL_LINES_URL = BASE + "?key7=" + urllib.parse.quote("所有线路输出到本页")
DEFAULT_LINES = ["京广高速线"]


def _http_get(url, timeout=30):
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def fetch_all_line_names():
    """从“所有线路”页解析全部线路名（去重）。"""
    text = _http_get(ALL_LINES_URL)
    names = re.findall(r'href=\?linename=([^"&>]+)', text)
    names = sorted(set(urllib.parse.unquote(x) for x in names))
    return names


def fetch_all_lines_summary():
    """从“所有线路”页解析每条线的总里程(km)，返回 {线名: 总里程}。

    该页一次请求即含全部线路的总里程，可作为增量抓取的变更指纹：
    远端某条线增删站点后其总里程必然变化 -> 仅重抓指纹不一致的线路。"""
    text = _http_get(ALL_LINES_URL)
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
        m = re.search(r'href=\?linename=([^"&>]+)', row)
        if not m:
            continue
        kms = re.findall(r"([\d,]+)km", row)
        if kms:
            out[urllib.parse.unquote(m.group(1))] = float(kms[0].replace(",", ""))
    return out


def fetch_line_mileage(line_name):
    """返回 [(站名, 距起始站累计 km), ...]（按页面顺序），解析失败返回 []。

    里程表为 12 列表格：途经车站(col2, <a href=?statinfo=站名>) 与
    距起始站里程(col8)/相邻站里程(col9, 形如 '274km')。逐行取首个 statinfo
    链接文本作为站名、取行内前两个 '数字km' 作为累计/相邻里程，跳过电报码(-开头)
    与车站编号(纯数字)等噪声链接。
    """
    url = BASE + "?linename=" + urllib.parse.quote(line_name)
    text = _http_get(url)

    idx = text.find("下面显示该线路的里程表")
    if idx == -1:
        print(f"  [警告] 未找到『{line_name}』里程表段落")
        return []
    seg = text[idx:]

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", seg, re.S)
    out = []
    for row in rows:
        m_name = re.search(r'<a href=\?statinfo=([^>]+)>([^<]+)</a>', row)
        if not m_name:
            continue
        name = m_name.group(2).strip()
        if name.startswith("-") or name.isdigit():
            continue
        kms = re.findall(r"([\d,]+)km", row)
        if len(kms) < 2:
            continue
        out.append((name, float(kms[0].replace(",", ""))))
    return out


def main():
    ap = argparse.ArgumentParser(description="抓取 jprailfan 线路每站累计里程")
    ap.add_argument("--lines", nargs="+", default=DEFAULT_LINES,
                    help="要抓取的线路名（与 jprailfan 一致，如 京广高速线）")
    ap.add_argument("--out", default="datasets/station_mileage.csv")
    ap.add_argument("--all-lines", action="store_true",
                    help="增量抓取全路网(指纹比对,仅新增/变更线路) -> network.json")
    ap.add_argument("--force", action="store_true",
                    help="忽略增量判定,强制重抓全部线路")
    ap.add_argument("--network-out", default="datasets/network.json")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条线路（0=全部）")
    ap.add_argument("--offset", type=int, default=0, help="从第 N 条开始处理")
    ap.add_argument("--delay", type=float, default=0.1, help="每条线路请求间隔(秒)")
    args = ap.parse_args()

    if args.all_lines:
        summary = fetch_all_lines_summary()
        names = sorted(summary)
        print(f"远端共 {len(names)} 条线路")

        net, meta = {}, {}
        if os.path.exists(args.network_out):
            with open(args.network_out, encoding="utf-8") as f:
                old = json.load(f)
            net = old.get("lines", {})
            meta = old.get("meta", {})

        # ---- 增量判定：新增 / 里程指纹变化 / 未变（跳过）----
        todo = []          # [(线名, 原因)]
        n_skip = 0
        now = datetime.now().isoformat(timespec="seconds")
        for ln in names:
            local = net.get(ln)
            if args.force:
                todo.append((ln, "强制"))
                continue
            if not local:                       # 本地无数据（新线或上次抓失败）
                todo.append((ln, "新增"))
                continue
            local_km = max(k for _, k in local)
            remote_km = summary[ln]
            if ln not in meta:
                # 旧格式文件无指纹：里程一致则视为未变并补指纹，不一致才重抓
                if abs(local_km - remote_km) <= 0.5:
                    meta[ln] = {"total_km": remote_km, "fetched_at": now}
                    n_skip += 1
                    continue
                todo.append((ln, f"里程不符({local_km:.0f}->{remote_km:.0f}km)"))
                continue
            if abs(meta[ln].get("total_km", -1.0) - remote_km) > 0.5:
                old_km = meta[ln].get("total_km", -1.0)
                todo.append((ln, f"更新({old_km:.0f}->{remote_km:.0f}km)"))
                continue
            n_skip += 1
        print(f"增量判定: 待抓 {len(todo)} 条 | 未变跳过 {n_skip} 条")

        todo = todo[args.offset:]
        if args.limit:
            todo = todo[:args.limit]

        done = 0
        for ln, reason in todo:
            try:
                seq = fetch_line_mileage(ln)
            except Exception as e:  # 单条失败不影响整体
                print(f"  [跳过] {ln}: {e}")
                seq = []
            net[ln] = [[s, float(k)] for s, k in seq]
            meta[ln] = {"total_km": summary.get(ln, max((k for _, k in seq), default=0.0)),
                        "fetched_at": now}
            done += 1
            if seq:
                print(f"  [{reason}] {ln}: {len(seq)} 站")
            time.sleep(args.delay)

        Path(args.network_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.network_out, "w", encoding="utf-8") as f:
            json.dump({"lines": net, "meta": meta}, f, ensure_ascii=False)
        print(f"已写出 {args.network_out}（{len(net)} 线 | 本次抓取 {done} | 未变 {n_skip}）")
        return

    # 单条/少量线路模式
    merged = {}
    for ln in args.lines:
        print(f"抓取线路: {ln}")
        seq = fetch_line_mileage(ln)
        print(f"  解析到 {len(seq)} 个站")
        for s, k in seq:
            merged[s] = k

    if not merged:
        raise SystemExit("没有任何站点里程，终止。")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["车站名", "累计km"])
        for name, v in merged.items():
            w.writerow([name, v])
    print(f"已写出 {args.out}（{len(merged)} 站）")


if __name__ == "__main__":
    main()
