#!/usr/bin/env python3
"""
根据输入股票代码，抓取最近约两个半月（日K）走势，
并在A股候选池中寻找K线形态最相似的股票。

数据源：东方财富公开 API（clist + kline）。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import json
import urllib.parse
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Referer": "https://quote.eastmoney.com/"}


@dataclass
class StockInfo:
    code: str
    name: str
    market: int  # f13

    @property
    def secid(self) -> str:
        return f"{self.market}.{self.code}"


def request_json(url: str, params: Dict, timeout: float = 10.0) -> Dict:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    req = urllib.request.Request(full_url, headers=HEADERS, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8"))
    if data.get("data") is None:
        raise ValueError(f"API返回为空: {data}")
    return data


def fetch_a_share_list() -> List[StockInfo]:
    """获取沪深主板/创业板/科创板候选列表。"""
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    result: List[StockInfo] = []
    page = 1
    while True:
        params = {
            "pn": page,
            "pz": 1000,
            "po": 1,
            "np": 1,
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": 2,
            "invt": 2,
            "fid": "f3",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            "fields": "f12,f14,f13",
        }
        data = request_json(url, params)["data"]
        diff = data.get("diff") or []
        if not diff:
            break

        for row in diff:
            code = str(row.get("f12", "")).strip()
            name = str(row.get("f14", "")).strip()
            market = int(row.get("f13", -1))
            if code and name and market in (0, 1):
                result.append(StockInfo(code=code, name=name, market=market))

        total = int(data.get("total", len(result)))
        if page * 1000 >= total:
            break
        page += 1
    return result


def fetch_kline(secid: str, lmt: int) -> List[Tuple[float, float, float, float, float]]:
    """返回 [(open, close, high, low, vol), ...]"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "klt": 101,  # 日K
        "fqt": 1,
        "lmt": lmt,
        "end": "20500000",
        "iscca": 1,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    data = request_json(url, params)["data"]
    klines = data.get("klines") or []
    out: List[Tuple[float, float, float, float, float]] = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 6:
            continue
        try:
            o = float(parts[1])
            c = float(parts[2])
            h = float(parts[3])
            l = float(parts[4])
            v = float(parts[5])
        except ValueError:
            continue
        if min(o, c, h, l) <= 0:
            continue
        out.append((o, c, h, l, v))
    return out


def build_feature(candles: Sequence[Tuple[float, float, float, float, float]]) -> List[float]:
    """构造K线形态特征并做标准化。"""
    feat: List[float] = []
    prev_close: Optional[float] = None

    for o, c, h, l, v in candles:
        body = (c - o) / o
        upper = (h - max(o, c)) / o
        lower = (min(o, c) - l) / o
        ret = 0.0 if prev_close is None else (c - prev_close) / prev_close
        vratio = 0.0 if v <= 0 else math.log1p(v)
        feat.extend((body, upper, lower, ret, vratio))
        prev_close = c

    if not feat:
        return []

    mean = sum(feat) / len(feat)
    var = sum((x - mean) ** 2 for x in feat) / len(feat)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return [(x - mean) / std for x in feat]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="查找最近两个半月K线形态相似股票（东方财富API）")
    p.add_argument("code", help="目标股票代码，如 600519")
    p.add_argument("--topn", type=int, default=10, help="输出最相似的前N只")
    p.add_argument(
        "--lookback-days",
        type=int,
        default=55,
        help="回看交易日数量（两个半月约55个交易日）",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=500,
        help="最大候选股票数量（越大越慢）",
    )
    p.add_argument("--workers", type=int, default=16, help="并发请求线程数")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    try:
        stocks = fetch_a_share_list()
    except Exception as e:
        print(f"获取A股列表失败，请检查网络或代理设置: {e}")
        return 1
    if not stocks:
        print("未获取到A股列表")
        return 1

    stock_map = {s.code: s for s in stocks}
    target = stock_map.get(args.code)
    if not target:
        print(f"找不到目标股票代码: {args.code}（需A股6位代码）")
        return 1

    try:
        target_kl = fetch_kline(target.secid, args.lookback_days)
    except Exception as e:
        print(f"拉取目标股票K线失败: {e}")
        return 1
    if len(target_kl) < args.lookback_days:
        print(f"目标股票历史数据不足: {len(target_kl)} < {args.lookback_days}")
        return 1
    target_feat = build_feature(target_kl[-args.lookback_days :])

    candidates = [s for s in stocks if s.code != target.code][: args.max_candidates]
    results: List[Tuple[float, StockInfo]] = []

    def worker(s: StockInfo) -> Optional[Tuple[float, StockInfo]]:
        try:
            kl = fetch_kline(s.secid, args.lookback_days)
            if len(kl) < args.lookback_days:
                return None
            feat = build_feature(kl[-args.lookback_days :])
            sim = cosine_similarity(target_feat, feat)
            if sim <= -0.5:
                return None
            return sim, s
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item in ex.map(worker, candidates):
            if item is not None:
                results.append(item)

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[: args.topn]

    print(f"目标股票: {target.code} {target.name}")
    print(f"比较区间: 最近 {args.lookback_days} 个交易日（日K，约两个半月）")
    print(f"候选数量: {len(candidates)}，有效比较: {len(results)}")
    print("\n最相似股票：")
    for i, (sim, s) in enumerate(top, 1):
        print(f"{i:>2}. {s.code} {s.name:<8}  相似度: {sim:.4f}")

    print(f"\n耗时: {time.time()-t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
