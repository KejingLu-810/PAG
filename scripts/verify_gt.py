import numpy as np
import faiss

# ----------------------------
# Configure paths (modify according to your actual paths)
# ----------------------------
data_path = "/mnt/mydata/datasets/glove1.2m/glove1.2m_base.fvecs"
query_path = "/mnt/mydata/datasets/glove1.2m/glove1.2m_query.fvecs"
gt_ivecs_path = "/mnt/mydata/datasets/glove1.2m/final_knn100_indices.ivecs"

K = 100  # Top-K

# ----------------------------
# Helper function: Read .fvecs files
# ----------------------------
def fvecs_read(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
                vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                yield vec
            except IndexError:
                break

def ivecs_read(filename):
    a = np.fromfile(filename, dtype='int32')
    d = a[0]  # The first number is K
    return a.reshape(-1, d + 1)[:, 1:]  # Skip the K header of each row

# ----------------------------
# Load data
# ----------------------------
print("Loading data...")
xb = np.vstack(list(fvecs_read(data_path))).astype('float32')
print(f"Data shape: {xb.shape}")  # Should be (1193514, 200)

print("Loading queries...")
xq = np.vstack(list(fvecs_read(query_path))).astype('float32')
print(f"Query shape: {xq.shape}")  # Should be (1000, 200)

print("Loading ground truth...")
gt = ivecs_read(gt_ivecs_path)
print(f"GT shape: {gt.shape}")     # Should be (1000, 100)

# ----------------------------
# Compute L2 nearest neighbors with FAISS (IndexFlatL2)
# ----------------------------
print("\n[1/2] Verifying with L2 distance...")
index_l2 = faiss.IndexFlatL2(xb.shape[1])
index_l2.add(xb)
D_l2, I_l2 = index_l2.search(xq[:1], K)  # Only search the first query

# ----------------------------
# Compute Inner Product with FAISS (normalize first → equivalent to cosine)
# ----------------------------
print("[2/2] Verifying with Cosine (IP on normalized vectors)...")
xb_norm = xb / np.linalg.norm(xb, axis=1, keepdims=True)
xq_norm = xq / np.linalg.norm(xq, axis=1, keepdims=True)

index_ip = faiss.IndexFlatIP(xb_norm.shape[1])
index_ip.add(xb_norm.astype('float32'))
D_ip, I_ip = index_ip.search(xq_norm[:1].astype('float32'), K)

# ----------------------------
# Compare with GT (first query)
# ----------------------------
gt_first = gt[0]  # Top-100 of the first query in your .ivecs file

print("\n" + "="*60)
print("Comparison for Query #0:")
print("="*60)

# Calculate intersection
inter_l2 = len(set(gt_first) & set(I_l2[0]))
inter_ip = len(set(gt_first) & set(I_ip[0]))

recall_l2 = inter_l2 / K
recall_ip = inter_ip / K

print(f"Your GT vs FAISS L2:   {inter_l2}/100 → Recall@100 = {recall_l2:.4f}")
print(f"Your GT vs FAISS IP:   {inter_ip}/100 → Recall@100 = {recall_ip:.4f}")

if recall_l2 > recall_ip:
    print("\nL2  Your GT is likely computed with L2 distance.")
elif recall_ip > recall_l2:
    print("\nCOS Your GT is likely computed with Cosine (IP) distance.")
else:
    print("\n❓ Inconclusive — both low.")

# Optional: Print comparison of top 10 IDs
print("\nTop-10 IDs in your GT:      ", gt_first[:10])
print("Top-10 IDs from FAISS L2:   ", I_l2[0][:10])
print("Top-10 IDs from FAISS IP:   ", I_ip[0][:10])