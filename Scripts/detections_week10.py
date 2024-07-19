import os
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from collections import Counter

# Define paths for video file, output file, and directory to save extracted frames.
video_file = "/data/home/kpatherya3/test/hmm/data/0001_vid.mp4"
output_file = "/data/home/kpatherya3/test/hmm/data/clips/example.npy"
clips_dir = "/data/home/kpatherya3/test/hmm/data/clips"

# Create the directory for saving clips if it does not exist.
if not os.path.exists(clips_dir):
    os.makedirs(clips_dir)

"""
Extract frames from the video at a given frame rate.
Args:
    video_path (str): Path to the video file.
    frame_rate (int): Frame rate to extract frames.
Returns:
    list: Extracted frames.
    list: Information about extracted frames.
"""
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

"""
Read all videos in a directory, extract frames from each video, and return all frames with their information.
Args:
    directory (str): Directory containing video files.
    frame_rate (int): Frame rate to extract frames.
Returns:
    list: All extracted frames.
    list: Information about all extracted frames.
"""
def read_videos_and_extract_frames_with_info(directory, frame_rate=30):
    all_frames = []
    all_frame_info = []
    video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    for video_path in video_files:
        frames, frame_info = extract_frames_with_info(video_path, frame_rate)
        all_frames.extend(frames)
        all_frame_info.extend(frame_info)
    return all_frames, all_frame_info

"""
Augment a given frame with random rotation and brightness adjustment to increase data variability.
Args:
    frame (numpy.ndarray): Input video frame.
Returns:
    numpy.ndarray: Augmented frame.
"""
def augment_frame(frame):
    angle = np.random.uniform(-15, 15)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    frame = cv2.warpAffine(frame, M, (w, h))
    brightness = np.random.uniform(0.8, 1.2)
    frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
    return frame

"""
Extract features from a given frame using a pre-trained CNN (VGG16).
Args:
    frame (numpy.ndarray): Input video frame.
Returns:
    numpy.ndarray: Flattened feature vector extracted from the frame.
"""
def extract_features_cnn(frame):
    model = VGG16(weights='imagenet', include_top=False)
    img = cv2.resize(frame, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

"""
Apply a sliding window to the features to create sequences of averaged features.
Args:
    features (numpy.ndarray): Input feature vectors.
    window_size (int): Size of the sliding window.
Returns:
    numpy.ndarray: Windowed feature vectors.
"""
def sliding_window(features, window_size=5):
    windowed_features = []
    for i in range(len(features) - window_size + 1):
        window = features[i:i+window_size]
        windowed_features.append(np.mean(window, axis=0))
    return np.array(windowed_features)

"""
Sampling function for the VAE to generate latent space representations.
Args:
    args (tuple): Mean and log variance of the latent space.
Returns:
    tensorflow.Tensor: Sampled latent space representation.
"""
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

### Main script to extract frames, augment and extract features, reduce dimensionality, train VAE, and detect anomalies. ###

"""
Extract frames from all videos in the specified directory.
Print the number of frames extracted.
"""
all_frames, all_frame_info = read_videos_and_extract_frames_with_info(clips_dir)
print(f"Number of frames extracted: {len(all_frames)}")

"""
Extract and augment features for each frame using the defined functions.
Convert the list of features to a numpy array and print its shape.
"""
features = []
for frame in all_frames:
    augmented_frame = augment_frame(frame)
    features.append(extract_features_cnn(augmented_frame))

features_array = np.array(features)
print(f"Shape of features_array: {features_array.shape}")

"""
Reduce the dimensionality of the features using Gaussian Random Projection and Incremental PCA.
Print the shape of the reduced feature array.
"""
initial_dim_reduction = GaussianRandomProjection(n_components=5000)
features_array_reduced = initial_dim_reduction.fit_transform(features_array)

ipca = IncrementalPCA(n_components=1000, batch_size=1000)
features_reduced = None
for i in range(0, len(features_array_reduced), 1000):
    batch = features_array_reduced[i:i+1000]
    if features_reduced is None:
        features_reduced = ipca.fit_transform(batch)
    else:
        features_reduced = np.vstack((features_reduced, ipca.transform(batch)))

print(f"Shape of features_reduced: {features_reduced.shape}")

# Define and compile the Variational Autoencoder (VAE) model.
input_img = Input(shape=(features_reduced.shape[1],))
x = Dense(128, activation='relu')(input_img)
x = Dense(64, activation='relu')(x)
z_mean = Dense(32)(x)
z_log_var = Dense(32)(x)
z = Lambda(sampling)([z_mean, z_log_var])
decoder = Dense(64, activation='relu')(z)
decoder = Dense(128, activation='relu')(decoder)
decoder = Dense(features_reduced.shape[1], activation='sigmoid')(decoder)

vae = Model(input_img, decoder)
vae.add_loss(0.5 * K.mean(K.square(z_mean) + K.exp(z_log_var) - z_log_var - 1, axis=-1))
vae.compile(optimizer='adam')

# Train the VAE with the reduced features, using a validation split for performance monitoring.
vae.fit(features_reduced, epochs=50, batch_size=256, validation_split=0.2)

"""
Detect anomalies by reconstructing features with the VAE and calculating reconstruction loss.
Flag anomalies based on the reconstruction loss percentile.
"""
reconstructed = vae.predict(features_reduced)
vae_loss = np.mean(np.square(features_reduced - reconstructed), axis=1)
vae_anomalies = vae_loss > np.percentile(vae_loss, 99)

"""
Use Isolation Forest for additional anomaly detection to complement VAE results.
Combine VAE and Isolation Forest predictions to identify final anomalies.
"""
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest_pred = iso_forest.fit_predict(features_reduced)

# Combine VAE and Isolation Forest predictions
combined_anomalies = np.logical_or(vae_anomalies, iso_forest_pred == -1)
anomalies = np.where(combined_anomalies)[0]

# Apply t-SNE to the reduced feature space for 2D visualization of anomalies
tsne = TSNE(n_components=2, random_state=42)
anomalous_features_2d = tsne.fit_transform(features_reduced[anomalies])

"""
Optimize DBSCAN clustering parameters using a grid search and silhouette score to find the best clustering configuration.
Print the best parameters found.
"""
param_grid = {'eps': [0.1, 0.5, 1.0], 'min_samples': [3, 5, 10]}
best_silhouette = -1
best_params = None

for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(**params)
    clusters = dbscan.fit_predict(anomalous_features_2d)
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(anomalous_features_2d, clusters)
        if score > best_silhouette:
            best_silhouette = score
            best_params = params

print(f"Best DBSCAN parameters: {best_params}")

# Apply DBSCAN with the best parameters to the 2D t-SNE features and visualize the clusters.
dbscan = DBSCAN(**best_params)
clusters = dbscan.fit_predict(anomalous_features_2d)

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(anomalous_features_2d[:, 0], anomalous_features_2d[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('DBSCAN Clusters of Anomalous Features')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.colorbar()
plt.show()

# Analyze and print the clusters of anomalies.
unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    cluster_indices = np.where(clusters == cluster)[0]
    print(f"Cluster {cluster}: Anomalous frames {anomalies[cluster_indices]}")

"""
Perform temporal analysis using a sliding window to detect sequences of anomalies.
Print the detected temporal anomalies.
"""
windowed_features = sliding_window(features_reduced)
windowed_anomalies = np.where(np.mean(combined_anomalies.reshape(-1, 5), axis=1) > 0.5)[0]

print("Temporal anomalies detected in windows:", windowed_anomalies)

# Map anomalies to their respective clips and print the frequency of anomalies in each clip.
anomalies_clips = anomalies // (30 * 5)  # 30 frames per second, 5 seconds per clip
clip_count = Counter(anomalies_clips)

for clip, frequency in clip_count.items():
    print(f"Clip {clip}: {frequency} occurrences")

"""
Save the results (anomalies, clusters, and frame information) to a file.
Print a message indicating that results have been saved.
"""
np.save(output_file, {
    'anomalies': anomalies,
    'clusters': clusters,
    'frame_info': all_frame_info
})

print(f"Results saved to {output_file}")