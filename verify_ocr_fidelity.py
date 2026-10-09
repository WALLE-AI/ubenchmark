#!/usr/bin/env python3
"""交叉校验: 确认 bench_ocr_e2e.py 压的是真实 OCR 负载，而不是空响应。

对压测数据集里的若干页跑一遍完整的两阶段流水线，把各 block 的识别结果按版面顺序
拼起来，和 PDF 文本层 (PyMuPDF get_text) 做字符级重合度比对。

用法:
  PY=/home/dataset-s3-0/gaojing/anaconda3/envs/docparserEnv/bin/python
  $PY verify_ocr_fidelity.py --dataset ocr_bench_dataset.json --pages 3
"""

import argparse
import asyncio
import json
import re
from collections import Counter

import aiohttp
import fitz
import numpy as np

from bench_ocr_e2e import (
    ADAPTERS,
    DEFAULT_TARGETS,
    SKIP_BLOCK_TYPES,
    build_request,
    crop_block,
    post_chat,
    prepare_layout_image,
    render_page,
    to_data_uri,
)

CJK_AND_ALNUM = re.compile(r"[一-鿿0-9a-zA-Z]")


def charset(text):
    return Counter(CJK_AND_ALNUM.findall(text))


def overlap_ratio(ocr_text, ref_text):
    """OCR 结果覆盖了文本层多少字符 (按多重集交集 / 文本层总量)。"""
    ref = charset(ref_text)
    if not ref:
        return None
    got = charset(ocr_text)
    inter = sum(min(n, got[c]) for c, n in ref.items())
    return inter / sum(ref.values())


async def parse_one_page(session, target, adapter, path, page_index, dpi, max_tokens, timeout):
    url = f"{target['base_url']}/v1/chat/completions"
    page_image = render_page(path, page_index, dpi)
    page_np = np.array(page_image)

    payload = build_request(target["model"], adapter.layout_prompt,
                            to_data_uri(prepare_layout_image(page_image)),
                            max_tokens, (0.0, 0.0))
    ok, _, text, _, _, err = await post_chat(session, url, payload, timeout)
    if not ok:
        return None, f"layout 请求失败: {err}"

    blocks = adapter.parse_layout(text)
    pieces = []
    for block in blocks:
        if block.type in SKIP_BLOCK_TYPES:
            continue
        crop = crop_block(page_image, page_np, block)
        if crop is None:
            continue
        payload = build_request(target["model"], adapter.prompt_for(block.type),
                                to_data_uri(crop), max_tokens,
                                adapter.penalties_for(block.type))
        ok_b, _, btext, _, _, _ = await post_chat(session, url, payload, timeout)
        if ok_b:
            pieces.append((block.type, btext))
    return pieces, None


async def main_async(args):
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    samples = []
    for entry in dataset["entries"]:
        samples.append((entry, entry["page_indices"][len(entry["page_indices"]) // 2]))
        if len(samples) >= args.pages:
            break

    targets = DEFAULT_TARGETS if not args.target else [
        t for t in DEFAULT_TARGETS if t["name"] in args.target
    ]

    async with aiohttp.ClientSession() as session:
        for entry, page_index in samples:
            doc = fitz.open(entry["path"])
            ref_text = doc.load_page(page_index).get_text()
            doc.close()
            ref_chars = sum(charset(ref_text).values())
            print(f"\n{'=' * 90}")
            print(f"{entry['file'][:70]}  p{page_index}  "
                  f"(PDF 文本层 {ref_chars} 个中英数字符)")

            for target in targets:
                adapter = ADAPTERS[target["adapter"]]
                pieces, err = await parse_one_page(
                    session, target, adapter, entry["path"], page_index,
                    entry["dpi"], args.max_tokens, args.timeout,
                )
                if err:
                    print(f"  [{target['name']}] {err}")
                    continue
                joined = "\n".join(t for _, t in pieces)
                ratio = overlap_ratio(joined, ref_text)
                empty = sum(1 for _, t in pieces if not t.strip())
                print(f"  [{target['name']}] {len(pieces)} 块, "
                      f"输出 {sum(charset(joined).values())} 字符, 空响应 {empty} 块, "
                      f"文本层覆盖率 "
                      f"{f'{ratio:.1%}' if ratio is not None else 'N/A (无文本层)'}")
                if args.show:
                    for btype, t in pieces[:args.show]:
                        print(f"      <{btype}> {t.strip()[:100]!r}")


def main():
    p = argparse.ArgumentParser(description="校验 OCR 压测负载的真实性")
    p.add_argument("--dataset", default="ocr_bench_dataset.json")
    p.add_argument("--pages", type=int, default=3, help="抽查几页，默认 3")
    p.add_argument("--target", action="append", help="只查某个服务名，可重复")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--show", type=int, default=3, help="每页打印前 N 块的识别结果")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
