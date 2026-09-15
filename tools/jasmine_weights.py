#!/usr/bin/env python3
"""jasmine 权重文件（JASMINE_WEIGHTS_V1）的读写。

格式：

    JASMINE_WEIGHTS_V1\\n
    f32\\n
    <count>\\n
    <name> <rows> <cols> <byte_offset>\\n     × count（offset 相对数据区起点）
    \\n                                        （空行分隔索引区与数据区）
    <二进制 float32，行优先>

所有矩阵都是 jasmine 原生布局（已由导出侧完成转置 / 拆分 / 权重绑定）。
C++ 侧的对应实现见 jas_weight_io.hpp 的 weight_file_t / weight_writer_t。
"""

import numpy as np
from pathlib import Path

MAGIC = "JASMINE_WEIGHTS_V1"
DTYPE = "f32"

class WeightWriter:
    """收集 (name, 2-D float32 ndarray)，最后写成 索引 + 数据 的单文件。"""

    def __init__(self):
        self.entries = []      # (name, rows, cols, byte_offset)
        self.chunks = []
        self.offset = 0        # 字节

    def add(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected 2-D array, got shape {arr.shape}")
        rows, cols = arr.shape
        if rows <= 0 or cols <= 0:
            raise ValueError(f"{name}: non-positive shape {arr.shape}")
        data = arr.tobytes()
        self.entries.append((name, rows, cols, self.offset))
        self.chunks.append(data)
        self.offset += len(data)

    def add_scalar(self, name, value):
        self.add(name, np.array([[value]], dtype=np.float32))

    def write(self, path):
        path = Path(path)
        header = [MAGIC, DTYPE, str(len(self.entries))]
        for name, rows, cols, off in self.entries:
            header.append(f"{name} {rows} {cols} {off}")
        header.append("")
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            for chunk in self.chunks:
                f.write(chunk)
        return path


class WeightReader:
    """按名字读取 tensor；自动校验 magic / dtype / shape / 边界。"""

    def __init__(self, path):
        with open(path, "rb") as f:
            self._blob = f.read()
        if not self._blob:
            raise ValueError(f"{path}: empty file")

        pos = 0

        def next_line():
            nonlocal pos
            if pos > len(self._blob):
                raise ValueError(f"{path}: truncated header")
            end = self._blob.find(b"\n", pos)
            if end < 0:
                end = len(self._blob)
            line = self._blob[pos:end].decode("ascii", "replace").rstrip("\r\n")
            pos = end + 1
            return line

        magic = next_line()
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r} (expected {MAGIC!r})")
        dtype = next_line()
        if dtype != DTYPE:
            raise ValueError(f"{path}: unsupported dtype {dtype!r} (expected {DTYPE!r})")

        count = int(next_line())
        self._index = {}
        for i in range(count):
            parts = next_line().split()
            if len(parts) != 4:
                raise ValueError(f"{path}: malformed manifest entry {i}: {parts}")
            name, rows, cols, off = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            if rows <= 0 or cols <= 0:
                raise ValueError(f"{path}: non-positive shape for {name!r}")
            self._index[name] = (rows, cols, off)

        next_line()             # 空行
        self._data_start = pos

    def __contains__(self, name):
        return name in self._index

    def __len__(self):
        return len(self._index)

    def names(self):
        return list(self._index)

    def shape(self, name):
        rows, cols, _ = self._entry(name)
        return rows, cols

    def get(self, name):
        """返回 float64 的 [rows, cols] ndarray（内部按 float32 存储）。"""
        rows, cols, off = self._entry(name)
        n = rows * cols
        start = self._data_start + off
        end = start + n * 4
        if end > len(self._blob):
            raise ValueError(f"{name}: data out of range")
        arr = np.frombuffer(self._blob, dtype="<f4", count=n, offset=start)
        return arr.reshape(rows, cols).astype(np.float64)

    def scalar(self, name):
        rows, cols = self.shape(name)
        if (rows, cols) != (1, 1):
            raise ValueError(f"{name}: expected 1x1, got {rows}x{cols}")
        return float(self.get(name)[0, 0])

    def _entry(self, name):
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"missing tensor {name!r}") from None
