import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Constants for directories and frame dimensions
DATA_DIR = "/content/drive/MyDrive/OMSCS/Cichlids/allVideos"
CSV_PATH = "/content/drive/MyDrive/OMSCS/Cichlids/ManualLabels.csv"
FRAME_WIDTH = 64
FRAME_HEIGHT = 64
NUM_FRAMES = 10

# Label mapping for the behaviors
LABEL_MAPPING = {'c': 1, 'f': 2, 'p': 3, 't': 4, 'b': 5, 'm': 6, 's': 7, 'x': 8, 'o': 9, 'd': 10}

def gather_video_info(data_dir):
    """
    Traverse the data directory to gather information about all video files.
    
    Args:
    data_dir (str): The root directory containing video files.
    
    Returns:
    dict: A dictionary where the keys are video names and values are dictionaries containing video details.
    """
    video_info = {}
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.mp4'):
                video_path = os.path.join(root, filename)
                video_name = os.path.basename(video_path)
                video_info[video_name] = {'name': video_name, 'location': video_path}
    return video_info

def print_video_info(video_info):
    """
    Print the total number of videos found and sample video information.
    
    Args:
    video_info (dict): Dictionary containing video details.
    """
    print(f"Total video files found: {len(video_info)}")
    print("Sample video information:")
    sample_info = list(video_info.items())[:3]  # Print only first 3 items
    for name, info in sample_info:
        print(f"- Name: {name}")
        print(f"  Location: {info['location']}")

def update_drive_location(df, video_info):
    """
    Update the DataFrame to include the drive location for each video based on the video information.
    
    Args:
    df (pd.DataFrame): DataFrame containing video metadata.
    video_info (dict): Dictionary containing video details.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'DriveLocation' column filled.
    """
    df['DriveLocation'] = None
    for i, video_name in enumerate(df['ClipName']):
        for j, (video_drive, info) in enumerate(video_info.items()):
            if video_name in video_drive:
                df.at[i, 'DriveLocation'] = info['location']
    return df

def create_smaller_df(df):
    """
    Create a smaller DataFrame with only selected columns.
    
    Args:
    df (pd.DataFrame): Original DataFrame containing video metadata.
    
    Returns:
    pd.DataFrame: Smaller DataFrame with selected columns.
    """
    selected_columns = ['ClipName', 'DriveLocation', 'ManualLabel']
    smaller_df = df[selected_columns]
    return smaller_df

def filter_files_to_process(non_null_df):
    """
    Filter the files to process based on size constraints.
    
    Args:
    non_null_df (pd.DataFrame): DataFrame with non-null 'DriveLocation'.
    
    Returns:
    list: List of dictionaries containing file paths and labels.
    """
    files_to_process = []
    for index, row in non_null_df.iterrows():
        file_path = row['DriveLocation']
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024:  # Check if file size is greater than 100 KB
            files_to_process.append({'file': row['DriveLocation'], 'label': row['ManualLabel']})
    return files_to_process

def improved_convlstm_model(frame_width, frame_height, num_frames, num_classes):
    """
    Define and return an improved ConvLSTM model for video classification.
    
    Args:
    frame_width (int): Width of the video frames.
    frame_height (int): Height of the video frames.
    num_frames (int): Number of frames in each video.
    num_classes (int): Number of output classes.
    
    Returns:
    keras.Sequential: Compiled ConvLSTM model.
    """
    model = Sequential([
        ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(num_frames, frame_height, frame_width, 3)),
        BatchNormalization(),

        ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True),
        BatchNormalization(),

        ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=False),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history, save_path):
    """
    Plot and save the training history of the model.
    
    Args:
    history (History): Training history object.
    save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def extract_frames(video_path, frame_width, frame_height, num_frames):
    """
    Extract and resize frames from a video file.
    
    Args:
    video_path (str): Path to the video file.
    frame_width (int): Desired width of the frames.
    frame_height (int): Desired height of the frames.
    num_frames (int): Number of frames to extract.
    
    Returns:
    np.array: Array of extracted frames.
    """
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
    """
    Create a TensorFlow dataset from the processed files.
    
    Args:
    files_to_process (list): List of dictionaries containing file paths and labels.
    frame_width (int): Width of the video frames.
    frame_height (int): Height of the video frames.
    num_frames (int): Number of frames in each video.
    batch_size (int): Batch size for training.
    
    Returns:
    tf.data.Dataset: TensorFlow dataset for training.
    """
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

    dataset = dataset.map(augment_frames, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Main execution starts here
video_info = gather_video_info(DATA_DIR)
print_video_info(video_info)

# Load and process the CSV file containing manual labels
df = pd.read_csv(CSV_PATH)
df = update_drive_location(df, video_info)
smaller_df = create_smaller_df(df)

# Print information about the processed DataFrame
non_null_count = smaller_df['DriveLocation'].notnull().sum()
total_rows = len(smaller_df)
percentage = (non_null_count / total_rows) * 100

print(f"Number of instances with non-null 'DriveLocation': {non_null_count}")
print(f"\nTotal number of rows in the smaller dataframe: {total_rows}")
print(f"\nPercentage of rows with non-null 'DriveLocation': {percentage:.2f}%")

# Filter files to process based on size constraints
non_null_df = smaller_df[smaller_df['DriveLocation'].notnull()]
files_to_process = filter_files_to_process(non_null_df)

print(f"Total files to process: {len(files_to_process)}")
print(f"\nPercentage of >100 KB files: {(len(files_to_process) / non_null_count * 100):.2f}%")

# Create the TensorFlow dataset
dataset = create_tf_dataset(files_to_process, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES)

# Print sample batch information
for batch in dataset.take(1):
    frames, labels = batch
    print(f"Frames shape: {frames.shape}, Labels shape: {labels.shape}")
    
# Split the dataset into training and testing sets
dataset_size = len(files_to_process)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

dataset = dataset.cache()
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size).take(test_size)

# Define and compile the ConvLSTM model
model = improved_convlstm_model(FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES, num_classes=len(LABEL_MAPPING))

# Mixed precision training setup
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Compile the model with Adam optimizer and categorical crossentropy loss
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler with warm-up
def lr_schedule(epoch):
    if epoch < 5:
        return 1e-3 * (epoch + 1) / 5
    return 1e-3 * tf.math.exp(0.1 * (10 - epoch))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model with training and validation datasets
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[lr_scheduler])

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Extract true labels and predictions for performance analysis
y_true = []
y_pred = []
for batch in test_dataset:
    frames, labels = batch
    predictions = model.predict(frames)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))
    
# Plot training history
plot_training_history(history, "/content/drive/MyDrive/OMSCS/Cichlids/training_history.png")

# Performance Analysis Functions
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save the confusion matrix.
    
    Args:
    y_true (list): List of true labels.
    y_pred (list): List of predicted labels.
    class_names (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("/content/drive/MyDrive/OMSCS/Cichlids/confusion_matrix.png")
    plt.close()
    
def plot_roc_curve(y_true, y_pred_proba, class_names):
    """
    Plot and save the ROC curve.
    
    Args:
    y_true (list): List of true labels.
    y_pred_proba (list): List of predicted probabilities.
    class_names (list): List of class names.
    """
    y_true_binary = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_binary.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    
    for i, color in zip(range(n_classes), plt.cm.rainbow(np.linspace(0, 1, n_classes))):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('/content/drive/MyDrive/OMSCS/Cichlids/roc_curve.png')
    plt.close()
    
def plot_precision_recall_curve(y_true, y_pred_proba, class_names):
    """
    Plot and save the precision-recall curve.
    
    Args:
    y_true (list): List of true labels.
    y_pred_proba (list): List of predicted probabilities.
    class_names (list): List of class names.
    """
    y_true_binary = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_binary.shape[1]
    
    precision = dict()
    recall = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary[:, i], y_pred_proba[:, i])
        
    plt.figure(figsize=(10, 8))
    
    for i, color in zip(range(n_classes), plt.cm.rainbow(np.linspace(0, 1, n_classes))):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'Precision-Recall curve of class {class_names[i]}')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig('/content/drive/MyDrive/OMSCS/Cichlids/pr_curve.png')
    plt.close()
    
# Performance Analysis
class_names = list(LABEL_MAPPING.keys())
y_pred_proba = model.predict(test_dataset)
y_pred = np.argmax(y_pred_proba, axis=1)

# Plot confusion matrix, ROC curve, and precision-recall curve
plot_confusion_matrix(y_true, y_pred, class_names)
plot_roc_curve(y_true, y_pred_proba, class_names)
plot_precision_recall_curve(y_true, y_pred_proba, class_names)

# Print classification report
classification_report_str = classification_report(y_true, y_pred, target_names=class_names)
print(classification_report_str)