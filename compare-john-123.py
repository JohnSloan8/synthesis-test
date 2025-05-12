import librosa
import librosa.display
# import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

# Paths to audio files
ground_truth_path = 'audio-files/123/01-correct.wav'
audio_paths = ['audio-files/123/02-correct.wav', 'audio-files/123/03-wrong.wav', 'audio-files/123/04-v-wrong.wav', 'audio-files/123/05-nonsense.wav']

# Load ground truth
y_gt, sr_gt = librosa.load(ground_truth_path)
mfcc_gt = librosa.feature.mfcc(y=y_gt, sr=sr_gt, n_mfcc=13)

# Load other audios
audios = []
mfccs = []
for path in audio_paths:
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    audios.append((y, sr))
    mfccs.append(mfcc)

# Function to warp target MFCC to ground truth
def warp_features_to_ground_truth(ref_mfcc, target_mfcc):
    ref_mfcc_t = ref_mfcc.T
    target_mfcc_t = target_mfcc.T

    distance, path = fastdtw(ref_mfcc_t, target_mfcc_t, dist=euclidean)
    aligned_target = []
    for ref_idx, target_idx in path:
        aligned_target.append(target_mfcc_t[target_idx])

    aligned_target = np.array(aligned_target)
    return aligned_target, path, distance

# Compare each audio to ground truth
for i, mfcc in enumerate(mfccs):
    aligned_mfcc, path, distance = warp_features_to_ground_truth(mfcc_gt, mfcc)

    # Align ground truth to same length for fair comparison
    gt_aligned = mfcc_gt.T[[ref for ref, _ in path]]

    # Compute frame-by-frame squared error
    frame_errors = np.mean((aligned_mfcc - gt_aligned) ** 2, axis=1)

    # Print summary
    mse = np.mean(frame_errors)
    print(f"\nAudio {i+2} vs Ground Truth:")
    print(f"DTW distance: {distance:.2f}")
    print(f"MSE after warping: {mse:.4f}")

    # Plot error per frame
    # plt.figure(figsize=(10, 4))
    # plt.plot(frame_errors)
    # plt.title(f"Frame-by-Frame Error: Ground Truth vs Audio {i+2}")
    # plt.xlabel('Frame Index')
    # plt.ylabel('Mean Squared Error per Frame')
    # plt.show()
