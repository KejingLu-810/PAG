# PAG

PAG (Projection-Augmented Graph) is a high-performance vector indexing and retrieval framework supporting multiple distance metrics (L2, Cosine). It features a robust build system and an integrated pipeline for both index construction and benchmark searching.

## 1. System Requirements

*   **Compiler**: GCC supporting C++17 (9.0+ recommended)
*   **Build Tools**: CMake (3.10+) and Make
*   **Scripting**: Python 3.6+ (for automated builds and benchmarking)
*   **Libraries**: 
    *   **OpenMP**: Required for multi-threaded index construction.
    *   *numactl*(optimal): Highly recommended for CPU core binding to ensure performance stability.

## 2. Project Construction

The project utilizes a Python-based build entry `build.py` to wrap the CMake workflow. It supports rapid switching of algorithmic features via macros.

### Basic Build
```bash
# Build all targets (L2, Cosine, Tools)
python3 build.py all

# Build only the L2 distance version
python3 build.py l2

# Build only the Cosine distance version
python3 build.py cos
```

## 3. Running Binaries

Compiled binaries are located in the `bin/` directory. Binaries adopt an integrated **"Build-or-Search"** logic:
*   If the index does not exist in the target path, the program starts **Indexing**.
*   If the index is already present, the program enters **Benchmark Search** mode.

### Command Line Arguments
Arguments must be passed in the following order:
1. `data_path`: Path to base vector file (.fbin)
2. `query_path`: Path to query vector file (.fbin)
3. `truth_path`: Path to ground truth file (.ibin)
4. `index_path`: Directory to store/load the index
5. `vecsize`: Total number of database vectors
6. `qsize`: Total number of query vectors
7. `dim`: Vector dimensionality
8. `topk`: Number of nearest neighbors to retrieve (K)
9. `efc`: Construction parameter `efConstruction`
10. `M`: Construction parameter `M` (neighbors per node)
11. `L`: Segmentation/Hierarchy parameter

### Manual Execution Example
```bash
# Set thread count for construction
export OMP_NUM_THREADS=32

# Run with numactl for performance isolation
[numactl --cpunodebind=0 --membind=0] ./bin/PAG_l2 \
    /path/to/base.fbin \
    /path/to/query.fbin \
    /path/to/gt.ibin \
    /path/to/index_dir \
    1000000 1000 128 100 2000 64 32
```

## 4. Automation Logic

The project includes a `run.sh` script to simplify the pipeline. You can modify the parameters inside `run.sh` to automate the indexing and searching process for specific datasets.


## 5. File Formats

*   **fbin**: High-performance binary float format. The first 8 bytes contain `N` (uint32) and `D` (uint32), followed by `N*D` `float32` values.
*   **ibin**: Integer index file. Same header format (N, D) as fbin, but the data section contains `int32` IDs.

---

## 6. Directory Structure
*   `bin/`: Final executables.
*   `build/`: Intermediate CMake build directory.
*   `l2/`, `cosine/`: Source code for specific distance metric variants.
*   `tools/`: Utility tools (e.g., `bin2vec`).
*   `build.py`: Unified build entry script.
*   `run.sh`: Helper script to run the indexing and searching pipeline.