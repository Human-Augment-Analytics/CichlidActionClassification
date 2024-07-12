import os
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
import matplotlib.pyplot as plt
from collections import Counter

# Define paths
video_file = "/data/home/kpatherya3/test/hmm/data/0001_vid.mp4"
output_file = "/data/home/kpatherya3/test/hmm/data/clips/example.npy"
clips_dir = "/data/home/kpatherya3/test/hmm/data/clips"

# Create the directory if it does not exist
if not os.path.exists(clips_dir):
    os.makedirs(clips_dir)

def extract_features(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_frame, 8, 1, method='uniform')
    return lbp.flatten()

def extract_frames_with_info(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_info = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            frames.append(frame)
            frame_info.append({
                'clip': os.path.basename(video_path),
                'frame_number': count
            })
        count += 1
    cap.release()
    return frames, frame_info

def read_videos_and_extract_frames_with_info(directory, frame_rate=30):
    all_frames = []
    all_frame_info = []
    video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    for video_path in video_files:
        frames, frame_info = extract_frames_with_info(video_path, frame_rate)
        all_frames.extend(frames)
        all_frame_info.extend(frame_info)
    return all_frames, all_frame_info

# Use the modified function now
all_frames, all_frame_info = read_videos_and_extract_frames_with_info(clips_dir)

# To verify the number of frames extracted
print(f"Number of frames extracted: {len(all_frames)}")

def extract_features(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_frame, 8, 1, method='uniform')
    return lbp.flatten()

frames = all_frames
features = []

for i in range(len(frames)):
    features.append(extract_features(frames[i]))

# Check if features are extracted correctly
print(f"Number of frames: {len(all_frames)}")
print(f"Number of feature vectors: {len(features) if 'features' in locals() else 'features not defined'}")

features_array = np.array(features)

# Check if features_array is created correctly
if 'features_array' in locals():
    print(f"Shape of features_array: {features_array.shape}")
else:
    print("features_array is not defined")

# Reduce initial dimensionality to 5000 features
initial_dim_reduction = GaussianRandomProjection(n_components=5000)
features_array_reduced = initial_dim_reduction.fit_transform(features_array)

# Then apply IncrementalPCA
ipca = IncrementalPCA(n_components=1000, batch_size=1000)
features_reduced = None
for i in range(0, len(features_array_reduced), 100):
    batch = features_array_reduced[i:i+1000]
    if features_reduced is None:
        features_reduced = ipca.fit_transform(batch)
    else:
        features_reduced = np.vstack((features_reduced, ipca.transform(batch)))

print(f"Shape of features_reduced: {features_reduced.shape if features_reduced is not None else 'None'}")

# Define the autoencoder model
input_img = Input(shape=(features_reduced.shape[1],))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(features_reduced.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the autoencoder
autoencoder.fit(features_reduced, features_reduced, epochs=50, batch_size=256, validation_split=0.2)

# Detect anomalies using the autoencoder
reconstructed = autoencoder.predict(features_reduced)
loss = np.mean(np.square(features_reduced - reconstructed), axis=1)
anomalies = np.where(loss > np.percentile(loss, 99.5))[0]  # Top .5% as anomalies

# Apply DBSCAN to the anomalies
dbscan = DBSCAN(eps=0.5, min_samples=5)
anomalous_features = features_reduced[anomalies]
clusters = dbscan.fit_predict(anomalous_features)

# Plot the clusters
plt.figure(figsize=(8, 3))
plt.scatter(np.arange(len(clusters)), clusters, c=clusters, cmap='viridis', marker='o')
plt.title('DBSCAN Clusters of Anomalous Features')
plt.xlabel('Anomaly Index')
plt.ylabel('Cluster')
plt.colorbar()
plt.show()

# Analyze the clusters
unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    cluster_indices = np.where(clusters == cluster)[0]
    print(f"Cluster {cluster}: Anomalous frames {anomalies[cluster_indices]}")

anomalies_cut = np.where(anomalies < len(all_frames))[0]
anomalies_div = anomalies / (30 * 5) # 30 frames per second, 5 seconds per clip
anomalies_clips = anomalies_div.astype(int)
print(anomalies_clips)

# Provided data
clip_number = np.array(anomalies_clips)

# Count the occurrences of each clip number
clip_count = Counter(clip_number)

# Print the frequency table
for clip, frequency in clip_count.items():
    print(f"Clip {clip}: {frequency} occurrences")