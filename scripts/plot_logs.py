import matplotlib.pyplot as plt
import os

def extract_data_from_log(log_path):
    """
    Extract recall (2nd column) and QPS (3rd column) from log file
    Skip non-data lines (e.g., command lines, time statistics, etc.)
    """
    recalls, qps = [], []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # At least 3 columns, and the first 3 fields are all numeric (scientific notation or decimals allowed)
            if len(parts) >= 3:
                try:
                    # Try to convert the 2nd and 3rd columns to float
                    r = float(parts[1])
                    q = float(parts[2])
                    recalls.append(r)
                    qps.append(q)
                except ValueError:
                    continue  # Skip non-numeric lines
    return recalls, qps

def plot_logs(logs_dict, save_path=None, show=True, title="Recall vs QPS", figsize=(10, 6)):
    """
    Plot Recall-QPS curves for multiple log files
    
    Args:
        logs_dict (dict): {label: log_file_path}
        save_path (str or None): Save path, e.g., "result.png"; no saving if None
        show (bool): Whether to call plt.show()
        title (str): Chart title
        figsize (tuple): Image size
    """
    plt.figure(figsize=figsize)

    for label, log_path in logs_dict.items():
        if not os.path.exists(log_path):
            print(f"Warning: File not found - {log_path}")
            continue
        recalls, qps = extract_data_from_log(log_path)
        if recalls and qps:
            plt.plot(recalls, qps, marker='o', markersize=3, linestyle='-', label=label)
        else:
            print(f"Warning: No valid data found in {log_path}")

    plt.xlabel('Recall@10')
    plt.ylabel('QPS')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the image (if path is specified)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show the image (optional)
    if show:
        plt.show()
    else:
        plt.close()  # Avoid memory accumulation


# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    logs_to_plot = {
        "fbin-stream": "/home/xxx/code/fast_graph/my-PAG/l2_glove1.2m_fbinread.log",
        "fvecs": "/home/xxx/code/fast_graph/my-PAG/l2_glove1.2m_fvecread.log",
        "fbin-full": "/home/xxx/code/fast_graph/my-PAG/l2_glove1.2m_fbinread_full.log"
    }

    # Save as PNG file (high DPI recommended)
    plot_logs(
        logs_dict=logs_to_plot,
        save_path="recall_vs_qps_glove.png",   # ← Specify save path
        show=False                             # ← Do not pop up window (suitable for server running)
    )