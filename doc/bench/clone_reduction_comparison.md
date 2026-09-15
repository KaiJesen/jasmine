# Clone reduction benchmark (Release)

Baseline: `clone_reduction_baseline.txt`（优化前，同机 Release）。

After: `clone_reduction_after.txt`。

| Benchmark | before (ns, median) | after (ns) | 变化 |
|---|--:|--:|---|
| BM_MatDot/256 | 4,364,896 | 4,368,418 | ≈ 持平 |
| BM_MatDot/512 | 33,858,800 | 33,957,535 | ≈ 持平 |
| BM_MhaForward/2/32/8 | 114,227 | 114,346 | ≈ 持平 |
| BM_MhaForward/4/64/16 | 783,019 | 782,899 | ≈ 持平 |
| BM_MhaForward/4/64/32 | — | 1,743,138 | 新增 |
| BM_MhaDecodeStep/4/64/32 | — | 645,238 | 新增（prefill=32 后单步 decode） |
| BM_MhaDecodeStep/4/64/128 | — | 550,558 | 新增（prefill=128 后单步 decode） |

说明：训练向 `BM_MhaForward` 短序列主要收益在层间 mat 移动、去掉重复物化，量级与噪声相当。推理 decode 主要去掉每步对全量 KV 的 `clone()`；`BM_MhaDecodeStep` 为新增探针，cache 越长节省越多（O(seq·d_head) 拷贝/步/头）。
