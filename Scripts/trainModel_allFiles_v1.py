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
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Constants; TODO: get all the videos in the allVideos folder, use your GDrive as reference for this
DATA_DIR = "/data/home/kpatherya3/test/allVideos"
CSV_PATH = "/data/home/kpatherya3/test/allVideos/ManualLabels.csv"
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
NUM_FRAMES = 64
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
    for i, (name, info) in enumerate(video_info.items()):
        print(f"- Name: {name}")
        print(f"  Location: {info['location']}")
        if i >= 2:
            break

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

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def custom_3d_cnn(frame_width, frame_height, num_frames, num_classes):
    inputs = keras.Input(shape=(num_frames, frame_height, frame_width, 3))
    
    # Normalize input
    x = layers.Lambda(lambda x: x / 255.0)(inputs)
    
    # Using an advanced 3D CNN architecture that resembles a ResNet
    def residual_block(x, filters, kernel_size=(3, 3, 3)):
        y = layers.Conv3D(filters, kernel_size, padding="same")(x)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv3D(filters, kernel_size, padding="same")(y)
        y = layers.BatchNormalization()(y)
        out = layers.Add()([x, y])
        return layers.LeakyReLU()(out)
    
    x = layers.Conv3D(64, (7, 7, 7), strides=(1, 2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding="same")(x)
    
    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters)
        x = residual_block(x, filters)
        x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    
    x = layers.Dense(512, activation="leaky_relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def extract_frames(video_path, frame_width, frame_height, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        frames.append(frame)

    cap.release()
    # If the video has fewer frames, pad with black frames
    while len(frames) < num_frames:
        frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
    return np.array(frames[:num_frames])

def create_dataset(files_to_process, frame_width, frame_height, num_frames):
    X = []
    y = []
    for file_info in files_to_process:
        video_path = file_info['file']
        label = file_info['label']
        frames = extract_frames(video_path, frame_width, frame_height, num_frames)
        X.append(frames)
        y.append(label)
    return np.array(X), np.array(y)

def augment_frames(X, datagen):
    X_augmented = []
    for frames in X:
        frames_augmented = []
        # Augment the entire set of frames for a video
        for batch in datagen.flow(np.array(frames), batch_size=len(frames), shuffle=False):
            frames_augmented = batch
            break  # Only take the first batch to prevent infinite loop
        X_augmented.append(frames_augmented)
    return np.array(X_augmented)

def main():
    video_info = gather_video_info(DATA_DIR)
    print_video_info(video_info)
    
    df = pd.read_csv(CSV_PATH)
    df = update_drive_location(df, video_info)
    
    smaller_df = create_smaller_df(df)
    
    non_null_count = smaller_df['DriveLocation'].notnull().sum()
    total_rows = len(smaller_df)
    percentage = (non_null_count / total_rows) * 100
    print(f"Number of instances with non-null 'DriveLocation': {non_null_count}")
    print(f"Total number of rows in the smaller dataframe: {total_rows}")
    print(f"Percentage of rows with non-null 'DriveLocation': {percentage:.2f}%")
    
    non_null_df = smaller_df[smaller_df['DriveLocation'].notnull()]
    files_to_process = filter_files_to_process(non_null_df)
    
    print(f"Total files to process: {len(files_to_process)}")
    print(f"Percentage of >100 KB files: {(len(files_to_process) / non_null_count * 100):.2f}%")
    
    X, y = create_dataset(files_to_process, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES)
    y = np.array([LABEL_MAPPING[label] for label in y])
    y = to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Adding brightness augmentation
        zoom_range=0.2,  # Adding zoom augmentation
        shear_range=0.2  # Adding shear augmentation
    )

    X_train_augmented = augment_frames(X_train, datagen)

    # Convert to float32 and normalize
    X_train_augmented = X_train_augmented.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_augmented, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(96).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(96).prefetch(tf.data.AUTOTUNE)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = custom_3d_cnn(FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES, num_classes=10)

    # Implement learning rate scheduler with warm-up
    def lr_schedule(epoch):
        if epoch < 5:
            return 1e-3 * (epoch + 1) / 5
        return 1e-3 * tf.math.exp(0.1 * (10 - epoch))

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        train_dataset, 
        epochs=100, 
        validation_split=0.2,  # Use 20% of training data for validation
        callbacks=[early_stopping, reduce_lr, lr_scheduler]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Plot training history
    plot_training_history(history)

    # Generate confusion matrix
    y_pred = model.predict(test_dataset)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    plot_confusion_matrix(y_true, y_pred, list(LABEL_MAPPING.keys()))

    # Print classification report
    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAPPING.keys())))
        
if __name__ == "__main__":
    main()