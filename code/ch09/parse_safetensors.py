"""09-01: 零成本结构观察——手动解析 safetensors 文件头（不加载张量、不碰 GPU）。

结构（huggingface/safetensors 官方规范）：
  [8 字节]  little-endian uint64 = JSON 头长度
  [N 字节]  UTF-8 JSON：张量名 -> {dtype, shape, data_offsets}
  [剩余]    纯二进制张量数据区（可 mmap，零拷贝）

本脚本只读前 8+header_bytes 字节，用来演示"加载前先看清单"。
用法：python parse_safetensors.py [path]   # 缺省用本机 Qwen2.5-0.5B
"""
import json
import math
import os
import struct
import sys
import collections

DEFAULT = ("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/"
           "060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors")


def main() -> None:
    path = os.path.realpath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))

    meta = header.pop("__metadata__", {})
    dtypes = collections.Counter(v["dtype"] for v in header.values())
    n_params = sum(math.prod(v["shape"]) for v in header.values())
    offs = [v["data_offsets"] for v in header.values()]
    contiguous = all(offs[i][1] == offs[i + 1][0] for i in range(len(offs) - 1))
    data_bytes = file_size - 8 - header_len

    print(json.dumps({
        "file": path,
        "file_MB": round(file_size / 1e6, 1),
        "header_bytes": header_len,
        "header_pct": round(100 * header_len / file_size, 4),
        "n_tensors": len(header),
        "dtypes": dict(dtypes),
        "total_params": n_params,
        "__metadata__": meta,
        "data_starts_at": 8 + header_len,
        "data_bytes": data_bytes,
        "data_equals_params_x_bpe": data_bytes == n_params * 2,  # BF16 全展平时应为 True
        "contiguous_offsets": contiguous,
        "first3": [(k, header[k]["shape"]) for k in list(header)[:3]],
        "last": (list(header)[-1], header[list(header)[-1]]["shape"]),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
