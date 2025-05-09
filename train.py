import os
import numpy as np
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === Config ===
DATA_DIR = "data_sample"
IMAGE_SIZE = 96
NUM_CLASSES = 26
EPOCHS = 15
BATCH_SIZE = 8
label_map = {char: idx for idx, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
reverse_label_map = {v: k for k, v in label_map.items()}


# === Load and preprocess data ===
def load_data():
    images = []
    labels = []

    for label in sorted(os.listdir(DATA_DIR)):  # ensure alphabetical folder order
        label_path = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_path) and label in label_map:
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                try:
                    img = Image.open(file_path).convert("RGBA")
                    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))  # white background
                    img = Image.alpha_composite(bg, img).convert("L")  # flatten alpha + grayscale
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                    img_array = np.array(img) / 255.0  # normalize
                    images.append(img_array)
                    labels.append(label_map[label])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    X = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# === Build the CNN model ===
def build_model():
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),  # Dropout layer with 25% probability
        # Second convolutional block
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.25),  # Dropout layer with 25% probability
        # Flattening and fully connected layers
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),  # Dropout layer with 50% probability
        Dense(26, activation="softmax")  # Output layer for 26 classes (A-Z)
    ])
    
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# === Main ===
if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    # Optional sanity check: view a sample
    sample_index = 0
    plt.imshow(X_train[sample_index].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title(f"Sample label: {reverse_label_map[np.argmax(y_train[sample_index])]}")
    plt.axis("off")
    plt.show()

    print("Training model...")
    model = build_model()
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[early_stop]
    )   
    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")


    preds = model.predict(X_test[:5])
    for i in range(5):
        plt.imshow(X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        plt.title(f"Predicted: {reverse_label_map[np.argmax(preds[i])]}, Actual: {reverse_label_map[np.argmax(y_test[i])]}")
        plt.axis("off")
        plt.show()
    print("Saving model as model.h5...")
    model.save("model.h5")

    # Save reverse label map for use in GUI
    with open("label_map.json", "w") as f:
        json.dump(reverse_label_map, f)

    print("Done!")
