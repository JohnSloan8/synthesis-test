import librosa
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Define datasets to process
datasets = [
    {
        "label": "Correct",
        "dir": "audio-files/amanda/",
        "prefix": "abairt-ceart"
    },
    {
        "label": "Wrong",
        "dir": "audio-files/amanda/",
        "prefix": "abairt-micheart"
    }
]

num_pairs = 5
target_duration = 3.0  # seconds

# Store frame-wise MSE data for plotting
frame_error_plots = {
    "Correct": [],
    "Wrong": []
}

# Normalize audio to target duration
def normalize_duration(y, sr, target_duration):
    current_duration = librosa.get_duration(y=y, sr=sr)
    target_length = int(target_duration * sr)

    if current_duration > target_duration:
        rate = current_duration / target_duration
        y = librosa.effects.time_stretch(y, rate=rate)
    elif current_duration < target_duration:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    
    return y

# Load MFCC with delta + delta-delta and preprocessing
def load_mfcc(path, n_mfcc=13):
    y, sr = librosa.load(path)
    y_trimmed, _ = librosa.effects.trim(y)
    y_normalized = normalize_duration(y_trimmed, sr, target_duration)

    mfcc = librosa.feature.mfcc(y=y_normalized, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.vstack([mfcc, delta, delta2])

# DTW alignment and MSE computation
def compare_mfccs(ref_mfcc, target_mfcc):
    ref_mfcc_t = ref_mfcc.T
    target_mfcc_t = target_mfcc.T

    distance, path = fastdtw(ref_mfcc_t, target_mfcc_t, dist=euclidean)

    aligned_target = np.array([target_mfcc_t[tgt] for _, tgt in path])
    aligned_ref = np.array([ref_mfcc_t[ref] for ref, _ in path])

    frame_errors = np.mean((aligned_target - aligned_ref) ** 2, axis=1)
    mse = np.mean(frame_errors)

    return distance, mse, frame_errors

# Main loop
for dataset in datasets:
    print(f"\n=== Comparing {dataset['label']} Files ===")
    mse_list = []

    for i in range(1, num_pairs + 1):
        idx = f"{i:02d}"
        dnc_path = f"{dataset['dir']}DNC-{dataset['prefix']}-{idx}.wav"
        piper_path = f"{dataset['dir']}PIPER-{dataset['prefix']}-{idx}.wav"

        mfcc_dnc = load_mfcc(dnc_path)
        mfcc_piper = load_mfcc(piper_path)

        distance, mse, frame_errors = compare_mfccs(mfcc_dnc, mfcc_piper)
        mse_list.append(mse)

        print(f"\nPair {idx}:")
        print(f"DTW distance: {distance:.2f}")
        print(f"MSE after warping: {mse:.4f}")

        # Save for plotting
        frame_error_plots[dataset['label']].append({
            "label": f"{dataset['label']} {idx}",
            "errors": frame_errors
        })

    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    print(f"\n>> Mean MSE for {dataset['label']} files: {mean_mse:.4f}")
    print(f">> Std  MSE for {dataset['label']} files: {std_mse:.4f}")

# === Plotting all 10 frame-wise MSE graphs ===
all_errors = [entry["errors"] for group in frame_error_plots.values() for entry in group]
global_min = min(np.min(err) for err in all_errors)
global_max = max(np.max(err) for err in all_errors)

plt.figure(figsize=(16, 12))
for i in range(num_pairs):
    # Left: Correct
    plt.subplot(num_pairs, 2, 2*i + 1)
    data = frame_error_plots["Correct"][i]
    plt.plot(data["errors"], label=data["label"])
    plt.title(data["label"])
    plt.ylim(global_min, global_max)
    plt.xlabel("Frame Index")
    plt.ylabel("MSE")
    plt.grid(True)

    # Right: Wrong
    plt.subplot(num_pairs, 2, 2*i + 2)
    data = frame_error_plots["Wrong"][i]
    plt.plot(data["errors"], label=data["label"], color='red')
    plt.title(data["label"])
    plt.ylim(global_min, global_max)
    plt.xlabel("Frame Index")
    plt.ylabel("MSE")
    plt.grid(True)

plt.suptitle("Frame-wise MSE per Pair (Correct Left, Wrong Right)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
