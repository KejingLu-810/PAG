# Projection-Augmented Graph (PAG)

PAG is a C++ and Python library for approximate nearest neighbor search. It supports static indexing and in-memory online insertion across L2, cosine, and maximum inner product search.

## Features

- Static `build`, `load`, `save`, single-query `search`, and batch search
- Online single-vector and batch `add` / `insert` after building an initial graph
- C++ public API and Python `import pag` interface
- L2, cosine, and MIPS metrics
- AVX-512 optimized CPU search kernels
- Command-line benchmark tool and reproducibility scripts

## Requirements

- Linux
- CMake 3.15+
- C++17 compiler
- OpenMP
- AVX-512 capable CPU
- Python 3.8+ for the Python package

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This builds:

```text
build/libpag_core.a
build/PAG
```

Install as a C++ library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/pag-install
cmake --build build -j$(nproc)
cmake --build build --target install
```

Install as a Python package:

```bash
python -m pip install .
```

After installation:

```python
import pag
```

## Quick Start

```cpp
#include "pag_index.h"

#include <random>
#include <vector>

int main() {
  const size_t count = 1024;
  const size_t dim = 32;
  std::vector<float> base(count * dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (float &value : base) {
    value = uniform(rng);
  }

  pag::BuildOptions build;
  build.index_path = "./pag_index";
  build.metric = pag::Metric::L2;
  build.max_search_k = 100;
  build.ef_construction = 100;
  build.projection_levels = 8;

  pag::Index index;
  index.build(base.data(), count, dim, build);

  pag::SearchOptions search;
  search.top_k = 10;
  search.ef_search = 100;

  std::vector<float> query(dim);
  auto results = index.search(query.data(), search);
}
```

Python:

```python
import numpy as np
import pag

base = np.random.random((100000, 128)).astype("float32")
queries = np.random.random((1000, 128)).astype("float32")

build = pag.BuildOptions()
build.index_path = "./pag_index"
build.metric = pag.Metric.L2
build.max_search_k = 100

index = pag.Index()
index.build(base, build)
ids, distances = index.search(queries, top_k=10, ef_search=100)
```

In C++, use `search_batch()` for row-major query batches and `add_batch()` / `insert_batch()` for online update blocks. In Python, passing a 2D numpy array to `search()` uses the same batch search path; `add_batch()` and `insert_batch()` are available for online indexes.

## Command-Line Tool

```bash
./build/PAG <base.fbin> <query.fbin> <truth.ibin> <index_dir> \
  <base_count> <query_count> <dim> <topk> \
  <ef_construction> <target_degree> <projection_levels> \
  [l2|cosine|mips] [max_search_k]
```

If `<index_dir>` exists, the tool loads the index and benchmarks search. Otherwise, it builds and saves a new index.

`topk` is the query result size for the current run. `max_search_k` is the largest `topk` the built index must support; the dataset scripts default it to `1000`.
`projection_levels` must be a positive multiple of `8`; the per-level projection code width is fixed by the 4-bit encoding format.

For headered `.fbin` and `.ibin` files, the command-line counts must match the file headers. Use the dataset scripts for benchmark runs.

MIPS builds use dataset order so that the same insertion semantics are available for online workloads.

## Benchmark Data

The benchmark datasets used by the scripts are hosted on Hugging Face:

```text
https://huggingface.co/datasets/ckadzh8/pag-benchmark-data
```

Download them into the repository root with:

```bash
python -m pip install "huggingface_hub[hf_xet]"
hf download ckadzh8/pag-benchmark-data --repo-type dataset \
  --include "data/*" --local-dir .
```

## Documentation

- [API guide](docs/api.md)
- [Build and packaging](docs/build.md)
- [Design overview](docs/design.md)
- [Benchmark summary](docs/benchmark.md)
- [Data format](docs/data_format.md)
- [Reproducibility](docs/reproducibility.md)

## Citation

If you use PAG, please cite:

```bibtex
@misc{lu2026pag,
  title = {Approximate Nearest Neighbor Search for Modern AI: A Projection-Augmented Graph Approach},
  author = {Lu, Kejing and Pan, Zhenpeng and Qin, Jianbin and Ishikawa, Yoshiharu and Xiao, Chuan},
  year = {2026},
  eprint = {2603.06660},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  doi = {10.48550/arXiv.2603.06660},
  url = {https://arxiv.org/abs/2603.06660}
}
```

Related paper:

```bibtex
@inproceedings{lu2026probabilistic,
  title = {Probabilistic Kernel Function for Fast Angle Testing},
  author = {Lu, Kejing and Xiao, Chuan and Ishikawa, Yoshiharu},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
  url = {https://openreview.net/forum?id=nCsF3Bsn2n}
}
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for the full license text.
