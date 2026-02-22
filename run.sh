#!/bin/bash

# --- PATH SETTING ---
BIN_DIR="/home/xxx/code/fast_graph/my-PAG/bin"
DATA_BASE="/data1/xxx/datasets/glove1.2m"
DATA_FILE="${DATA_BASE}/glove1.2m_base.fbin"
QUERY_FILE="${DATA_BASE}/glove1.2m_query.fbin"
GT_FILE="${DATA_BASE}/final_knn1000_indices.ibin"

# --- Exp Params. ---
ALGO="PAG_l2"        # choose: PAG_l2, PAG_l2_wopes
EFC=2000
M=64
L=32
TOPK=100
THREADS=112

# --- Dataset Info ---
N_VEC=1193514
QN_VEC=1000
DIM=200

# --- generate the dir of index automatically ---
IDX_DIR="/data/xxx/index/test_${ALGO}_${M}_${EFC}"
# clean index path
if [ -d "$IDX_DIR" ]; then
    rm -rf "$IDX_DIR"
fi
# makesure the index_path exists
mkdir -p "$(dirname "$IDX_DIR")"
# cd ${IDX_DIR}

echo "------------------------------------------------"
echo "Algorithm:  ${ALGO}"
echo "Dataset:    glove1.2m"
echo "Threads:    ${THREADS}"
echo "Index Path: ${IDX_DIR}"
echo "------------------------------------------------"

export OMP_NUM_THREADS=${THREADS}

start_time=$(date +%s.%N)

# --- Execute ---
"${BIN_DIR}/${ALGO}" \
"${DATA_FILE}" \
"${QUERY_FILE}" \
"${GT_FILE}" \
"${IDX_DIR}" \
"${N_VEC}" \
"${QN_VEC}" \
"${DIM}" \
"${TOPK}" \
"${EFC}" \
"${M}" \
"${L}"

end_time=$(date +%s.%N)

build_time=$(echo "$end_time - $start_time" | bc)

echo "------------------------------------------------"
echo "Execution Completed."
echo "Total Build Time (including loading): ${build_time} seconds"
echo "------------------------------------------------"

export OMP_NUM_THREADS=${1}

start_time=$(date +%s.%N)

"${BIN_DIR}/${ALGO}" \
"${DATA_FILE}" \
"${QUERY_FILE}" \
"${GT_FILE}" \
"${IDX_DIR}" \
"${N_VEC}" \
"${QN_VEC}" \
"${DIM}" \
"${TOPK}" \
"${EFC}" \
"${M}" \
"${L}"

end_time=$(date +%s.%N)

search_time=$(echo "$end_time - $start_time" | bc)

echo "------------------------------------------------"
echo "Execution Completed."
echo "Total search Time (including loading): ${search_time} seconds"
echo "------------------------------------------------"