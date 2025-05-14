import librosa
import numpy as np
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

# Function to compute MFCCs with silence trimming
def load_mfcc(path, n_mfcc=13):
    y, sr = librosa.load(path)
    y_trimmed, _ = librosa.effects.trim(y)  # Trim leading/trailing silence
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Function to warp and compare
def compare_mfccs(ref_mfcc, target_mfcc):
    ref_mfcc_t = ref_mfcc.T
    target_mfcc_t = target_mfcc.T

    distance, path = fastdtw(ref_mfcc_t, target_mfcc_t, dist=euclidean)

    aligned_target = np.array([target_mfcc_t[tgt] for _, tgt in path])
    aligned_ref = np.array([ref_mfcc_t[ref] for ref, _ in path])

    frame_errors = np.mean((aligned_target - aligned_ref) ** 2, axis=1)
    mse = np.mean(frame_errors)

    return distance, mse

# Main loop
for dataset in datasets:
    print(f"\n=== Comparing {dataset['label']} Files ===")
    mse_list = []

    for i in range(1, num_pairs + 1):
        idx = f"{i:02d}"
        dnc_path = f"{dataset['dir']}DNC-{dataset['prefix']}-{idx}.wav"
        piper_path = f"{dataset['dir']}PIPER-{dataset['prefix']}-{idx}.wav"

        # Load MFCCs (with silence trimmed)
        mfcc_dnc = load_mfcc(dnc_path)
        mfcc_piper = load_mfcc(piper_path)

        # Compare
        distance, mse = compare_mfccs(mfcc_dnc, mfcc_piper)
        mse_list.append(mse)

        # Output
        print(f"\nPair {idx}:")
        print(f"DTW distance: {distance:.2f}")
        print(f"MSE after warping: {mse:.4f}")

    # Compute statistics
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    print(f"\n>> Mean MSE for {dataset['label']} files: {mean_mse:.4f}")
    print(f">> Std  MSE for {dataset['label']} files: {std_mse:.4f}")
