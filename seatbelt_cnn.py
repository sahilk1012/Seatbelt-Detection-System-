import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             precision_score, recall_score, confusion_matrix)
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — change paths here if your folder structure is different
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_BELT_PATH         = 'dataset/Seat_Belt2/Train/Seat_Belt/'
TRAIN_NO_BELT_PATH      = 'dataset/Seat_Belt2/Train/WithoutSeat_Belt/'
TEST_BELT_PATH          = 'dataset/Seat_Belt2/Test/Seat_Belt/'
TEST_NO_BELT_PATH       = 'dataset/Seat_Belt2/Test/WithoutSeat_Belt/'

IMAGE_SIZE   = (128, 128)   # resize all images to this
EPOCHS       = 15
BATCH_SIZE   = 32
MODEL_SAVE   = 'seatbelt_model.h5'   # trained model saved here


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Check dataset exists
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset():
    for path in [TRAIN_BELT_PATH, TRAIN_NO_BELT_PATH,
                 TEST_BELT_PATH,  TEST_NO_BELT_PATH]:
        if not os.path.exists(path):
            print(f"\n❌ Folder not found: {path}")
            print("\nPlease download the dataset from:")
            print("https://www.kaggle.com/datasets/yehiahassanain/seat-belt2")
            print("\nExtract it so the folder structure is:")
            print("  dataset/Seat_Belt2/Train/Seat_Belt/")
            print("  dataset/Seat_Belt2/Train/WithoutSeat_Belt/")
            print("  dataset/Seat_Belt2/Test/Seat_Belt/")
            print("  dataset/Seat_Belt2/Test/WithoutSeat_Belt/")
            exit(1)
    print("✅ Dataset found.")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Load and convert images to numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_images(image_dir, target_size=IMAGE_SIZE):
    """Read all images from a folder, resize and return as numpy arrays."""
    data = []
    files = [f for f in os.listdir(image_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"  Loading {len(files)} images from {image_dir} ...")

    for img_file in files:
        img_path = os.path.join(image_dir, img_file)
        try:
            with Image.open(img_path) as image:
                image = image.resize(target_size)
                image = image.convert('RGB')
                data.append(np.array(image))
        except Exception as e:
            print(f"  ⚠️  Skipping {img_file}: {e}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Build CNN model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_shape=(128, 128, 3), num_classes=2):
    model = keras.Sequential([
        # Block 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Classifier head
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Plot training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("\n📊 Training curves saved as training_curves.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Predict a single image and show result
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(model, image_path):
    """Run prediction on one image and print the result."""
    if not os.path.exists(image_path):
        print(f"  ⚠️  Image not found: {image_path}")
        return

    input_image = mpimg.imread(image_path)

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(input_image)
    plt.axis('off')
    plt.title(os.path.basename(image_path))
    plt.show()

    # Preprocess
    resized  = cv2.resize(input_image, IMAGE_SIZE)
    scaled   = resized / 255.0
    reshaped = np.reshape(scaled, [1, 128, 128, 3])

    # Predict
    prediction   = model.predict(reshaped, verbose=0)
    pred_label   = np.argmax(prediction)
    confidence   = prediction[0][pred_label] * 100

    if pred_label == 0:
        print(f"  ✅ SEATBELT DETECTED       (confidence: {confidence:.1f}%)")
    else:
        print(f"  ❌ NO SEATBELT DETECTED    (confidence: {confidence:.1f}%)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_metrics(model, x_test_scaled, y_test, y_test_onehot):
    print("\n" + "="*50)
    print("  MODEL EVALUATION")
    print("="*50)

    # Loss and accuracy from Keras
    loss, accuracy = model.evaluate(x_test_scaled, y_test_onehot, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")

    # Sklearn metrics
    y_pred        = model.predict(x_test_scaled, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)

    mse       = mean_squared_error(y_test, y_pred_labels)
    mae       = mean_absolute_error(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels, zero_division=0)
    recall    = recall_score(y_test, y_pred_labels, zero_division=0)
    conf_mat  = confusion_matrix(y_test, y_pred_labels)

    print(f"\n  MSE           : {mse:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {conf_mat}")
    print("="*50)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n🚗 Seat Belt Detection — CNN Training Pipeline")
    print("="*50)

    # ── 1. Check dataset ──────────────────────────────────────────────────────
    check_dataset()

    # ── 2. Load images ────────────────────────────────────────────────────────
    print("\n📂 Loading training images...")
    data_belt    = load_images(TRAIN_BELT_PATH)
    data_nobelt  = load_images(TRAIN_NO_BELT_PATH)

    print(f"\n  Images with seatbelt    : {len(data_belt)}")
    print(f"  Images without seatbelt : {len(data_nobelt)}")

    # Labels: 0 = seatbelt, 1 = no seatbelt
    labels = [0] * len(data_belt) + [1] * len(data_nobelt)
    data   = data_belt + data_nobelt

    # ── 3. Prepare arrays ─────────────────────────────────────────────────────
    X = np.array(data)
    Y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    # Normalise pixel values to [0, 1]
    x_train_scaled = x_train / 255.0
    x_test_scaled  = x_test  / 255.0

    # One-hot encode labels
    num_classes       = 2
    y_train_onehot    = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_onehot     = tf.keras.utils.to_categorical(y_test,  num_classes)

    print(f"\n  Train samples : {len(x_train)}")
    print(f"  Test samples  : {len(x_test)}")

    # ── 4. Build and train ────────────────────────────────────────────────────
    print("\n🏋️  Building CNN model...")
    model = build_model()
    model.summary()

    print(f"\n🚀 Training for {EPOCHS} epochs...")
    history = model.fit(
        x_train_scaled, y_train_onehot,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # ── 5. Plot training curves ───────────────────────────────────────────────
    plot_training(history)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    evaluate_metrics(model, x_test_scaled, y_test, y_test_onehot)

    # ── 7. Save model ─────────────────────────────────────────────────────────
    model.save(MODEL_SAVE)
    print(f"\n💾 Model saved as: {MODEL_SAVE}")

    # ── 8. Test on sample images ──────────────────────────────────────────────
    print("\n🔍 Running predictions on sample test images...")

    # Collect up to 5 images from each test folder
    sample_images = []
    for folder in [TEST_BELT_PATH, TEST_NO_BELT_PATH]:
        if os.path.exists(folder):
            files = [os.path.join(folder, f)
                     for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sample_images.extend(files[:5])

    for img_path in sample_images:
        predict_image(model, img_path)

    print("\n✅ Done! Check training_curves.png for the loss/accuracy plots.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
