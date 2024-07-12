import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Constants
DATA_DIR = "/content/drive/MyDrive/OMSCS/Cichlids/allVideos"
CSV_PATH = "/content/drive/MyDrive/OMSCS/Cichlids/ManualLabels.csv"
FRAME_WIDTH = 64
FRAME_HEIGHT = 64
NUM_FRAMES = 10

"""
1 c BuildScoop
2 f FeedScoop
3 p BuildSpit
4 t FeedSpit
5 b BuildMultiple
6 m FeedMultiple
7 s Spawn
8 x NoFishOther
9 o FishOther
10 d DropSand
"""
LABEL_MAPPING = {'c': 1, 'f': 2, 'p': 3, 't': 4, 'b': 5, 'm': 6, 's': 7, 'x': 8, 'o': 9, 'd': 10}

def gather_video_info(data_dir):
    video_info = {}
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.mp4'):
                video_path = os.path.join(root, filename)
                video_name = os.path.basename(video_path)
                video_info[video_name] = {'name': video_name, 'location': video_path}
    return video_info

def print_video_info(video_info):
    print(f"Total video files found: {len(video_info)}")
    print("Sample video information:")
    sample_info = list(video_info.items())[:3]  # Print only first 3 items
    for name, info in sample_info:
        print(f"- Name: {name}")
        print(f"  Location: {info['location']}")

def update_drive_location(df, video_info):
    df['DriveLocation'] = None
    for i, video_name in enumerate(df['ClipName']):
        for j, (video_drive, info) in enumerate(video_info.items()):
            if video_name in video_drive:
                df.at[i, 'DriveLocation'] = info['location']
    return df

def create_smaller_df(df):
    selected_columns = ['ClipName', 'DriveLocation', 'ManualLabel']
    smaller_df = df[selected_columns]
    return smaller_df

def filter_files_to_process(non_null_df):
    files_to_process = []
    for index, row in non_null_df.iterrows():
        file_path = row['DriveLocation']
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024:
            files_to_process.append({'file': row['DriveLocation'], 'label': row['ManualLabel']})
    return files_to_process

def simple_convlstm_model(frame_width, frame_height, num_frames, num_classes):
    model = Sequential([
        # Remove the Lambda layer if you already provide the correct input shape
        ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(num_frames, frame_height, frame_width, 3)),
        BatchNormalization(epsilon=1e-5, momentum=0.9),

        ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True),
        BatchNormalization(epsilon=1e-5, momentum=0.9),

        ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=False),
        BatchNormalization(epsilon=1e-5, momentum=0.9),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Train Loss')
    #plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    #plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(save_path)
    plt.close()

def extract_frames(video_path, frame_width, frame_height, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames < num_frames:
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (frame_width, frame_height))
                frames.append(frame)
            else:
                frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (frame_width, frame_height))
                frames.append(frame)
            else:
                frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

    cap.release()

    return np.array(frames[:num_frames])

def create_tf_dataset(files_to_process, frame_width, frame_height, num_frames, batch_size=32):
    def generator():
        for file_info in files_to_process:
            video_path = file_info['file']
            label = LABEL_MAPPING[file_info['label']]
            frames = extract_frames(video_path, frame_width, frame_height, num_frames)
            yield frames, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(num_frames, frame_height, frame_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=len(LABEL_MAPPING))))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

video_info = gather_video_info(DATA_DIR)
print_video_info(video_info)

df = pd.read_csv(CSV_PATH)
df = update_drive_location(df, video_info)
smaller_df = create_smaller_df(df)

non_null_count = smaller_df['DriveLocation'].notnull().sum()
total_rows = len(smaller_df)
percentage = (non_null_count / total_rows) * 100

print(f"Number of instances with non-null 'DriveLocation': {non_null_count}")
print(f"\nTotal number of rows in the smaller dataframe: {total_rows}")
print(f"\nPercentage of rows with non-null 'DriveLocation': {percentage:.2f}%")

non_null_df = smaller_df[smaller_df['DriveLocation'].notnull()]
files_to_process = filter_files_to_process(non_null_df)

print(f"Total files to process: {len(files_to_process)}")
print(f"\nPercentage of >100 KB files: {(len(files_to_process) / non_null_count * 100):.2f}%")

# Create the tf.data.Dataset
dataset = create_tf_dataset(files_to_process, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES)

# Print sample batch information
for batch in dataset.take(1):
    frames, labels = batch
    print(f"Frames shape: {frames.shape}, Labels shape: {labels.shape}")
    
# Create datasets for training and testing
dataset_size = len(files_to_process)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

dataset = dataset.cache()
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size).take(test_size)

# Define the model
model = simple_convlstm_model(FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES, num_classes=len(LABEL_MAPPING))

# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

# Implement learning rate scheduler with warm-up
def lr_schedule(epoch):
    if epoch < 5:
        return 1e-3 * (epoch + 1) / 5
    return 1e-3 * tf.math.exp(0.1 * (10 - epoch))

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    callbacks=[lr_scheduler]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Extract true labels and predictions
y_true = []
y_pred = []

for batch in test_dataset:
    frames, labels = batch
    predictions = model.predict(frames)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))
    
# Plot training history
plot_training_history(history, "/content/drive/MyDrive/OMSCS/Cichlids/training_history.png")