import warnings ; warnings.filterwarnings('ignore')
import numpy as np
import scipy.ndimage.filters
import hmmlearn
from hmmlearn import hmm
import cv2
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

""" Define parameters """
mean_window = 120           # How many seconds does it take to calculate means for filtering out the noisy data
mean_cushion = 7.5          # Grayscale change in pixel value for filtering out large pixel changes for HMM analysis
hmm_window = 10             # Used for reducing the number of states for HMM calculation
seconds_to_change = 1800    # Used to determine the transition expectation from one state to another (i.e., how many manipulations occur)
non_transition_bins = 2     # Parameter to prevent small changes in the state
std = 100                   # Standard deviation of pixel data in HMM analysis

# Define paths
video_file = "/data/home/kpatherya3/test/hmm/data/0001_vid.mp4"
output_file = "/data/home/kpatherya3/test/hmm/data/clips/example.npy"
clips_dir = "/data/home/kpatherya3/test/hmm/data/clips"

# Create the dictionary if it does not exist
if not os.path.exists(clips_dir):
    os.makedirs(clips_dir)
    
def extract_frames(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def read_videos_and_extract_frames(directory, frame_rate=30):
    all_frames = []
    video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    for video_path in video_files:
        frames = extract_frames(video_path, frame_rate)
        all_frames.extend(frames)
    return all_frames

directory = clips_dir
all_frames = read_videos_and_extract_frames(directory)

# To verify the number of frames extracted
print(f"Number of frames extracted: {len(all_frames)}")

from scipy.ndimage import uniform_filter, median_filter
from skimage.feature import local_binary_pattern
from scipy.interpolate import griddata
import numpy as np

def interpolate_noisy_pixels(frame, noisy_pixels):
    # Convert the noisy_pixels to mask for OpenCV (invert for non-noisy pixels)
    mask = np.logical_not(noisy_pixels)
    # Use OpenCV's bilinear interpolation with boundary handling
    interpolated_frame = cv2.filter2D(frame.astype('float32'), -1, mask.astype('float32'), borderType=cv2.BORDER_REPLICATE)
    # Fill the remaining noisy pixels (if any) with the original value (or another strategy)
    interpolated_frame[noisy_pixels] = frame[noisy_pixels]
    return interpolated_frame.astype('uint8')

def extract_features(frame):
    # Extract LBP features from the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_frame, 8, 1, method='uniform')
    return lbp.flatten()

frames = all_frames  # Assuming `all_frames` is a list or numpy array of frames
filtered_frames = np.empty_like(frames)
threshold_multiplier = 3.82  # Adjust this value to change sensitivity
noisy_indices = []
features = []

for i in range(len(frames)):
    frame = frames[i]
    filtered_frame = uniform_filter(frame, size=3, mode='reflect').astype('uint8')
    frame_std = np.std(frame)
    noisy_pixels = np.abs(frame - filtered_frame) > threshold_multiplier * frame_std
    if np.any(noisy_pixels):
        #print(f"Frame {i} has noisy data.")
        frames[i] = interpolate_noisy_pixels(frame, noisy_pixels)  # Interpolate noisy pixels in the frame
        noisy_indices.append(i)
    filtered_frames[i] = filtered_frame
    features.append(extract_features(frames[i]))  # Extract features for each frame

if noisy_indices:
    print("Frames with noisy data:", noisy_indices)
else:
    print("No frames with noisy data found.")
    
from sklearn.decomposition import IncrementalPCA

features_array = np.array(features)

# Initialize the Incremental PCA model
ipca = IncrementalPCA(n_components=0.8, batch_size=100)

# Fit and transform in batches
features_reduced = None
for i in range(0, len(features_array), 100):
    batch = features_array[i:i+100]
    if features_reduced is None:
        features_reduced = ipca.fit_transform(batch)
    else:
        features_reduced = np.vstack((features_reduced, ipca.transform(batch)))

# Define the HMM model with adjusted parameters
n_components = 5    # Number of states in HMM, adjust based on use case
model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=50)

# Train the HMM model
model.fit(features_reduced)

# Predict the states for each frame (example)
predicted_states = model.predict(features_reduced)
print("Predicted states for each frame:\n", predicted_states)

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Apply DBSCAN to the predicted states
dbscan = DBSCAN(eps=0.5, min_samples=5)
predicted_states = predicted_states.reshape(-1, 1)  # Reshape for DBSCAN
clusters = dbscan.fit_predict(predicted_states)

# Plot the clusters
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(clusters)), clusters, c=clusters, cmap='viridis', marker='o')
plt.title('DBSCAN Clusters of Predicted States')
plt.xlabel('Frame Index')
plt.ylabel('Cluster')
plt.colorbar()
plt.show()

# Analyze the clusters
unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    cluster_indices = np.where(clusters == cluster)[0]
    print(f"Cluster {cluster}: {cluster_indices}")