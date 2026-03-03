import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 0. General settings
# ============================================================
OUTPUT_DIR = "cnn_homework_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

# Small helper for timestamped filenames
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


# ============================================================
# Helper: save detailed results (metrics, cm, predictions)
# ============================================================
def save_results(model_name, y_true, y_pred, class_names, output_dir):
    """
    Saves:
      - classification report (.txt)
      - predictions (true, pred) (.csv)
      - confusion matrix image (.png)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # 2. Predictions CSV
    pred_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    })
    pred_csv_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # 3. Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im)

    # For many classes (like CIFAR-100) we don't label all ticks with names,
    # just use indices to keep it readable.
    num_classes = len(class_names)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))

    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"\nSaved detailed results for {model_name}:")
    print(" -", report_path)
    print(" -", pred_csv_path)
    print(" -", cm_path)


# ============================================================
# 1. Custom CNN on Fashion-MNIST
# ============================================================
def run_fashion_mnist_cnn():
    print("\n=== Fashion-MNIST: Custom CNN ===")

    # -----------------------------
    # 1.1 Load data
    # -----------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    num_classes = 10
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # -----------------------------
    # 1.2 Build custom CNN model
    # -----------------------------
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    # -----------------------------
    # 1.3 Compile & train
    # -----------------------------
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train_cat,
        epochs=10,              # You can increase if you have time/GPU
        batch_size=128,
        validation_split=0.1,
        verbose=2,
    )

    # -----------------------------
    # 1.4 Evaluate
    # -----------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Fashion-MNIST Test accuracy: {test_acc:.4f}")
    print(f"Fashion-MNIST Test loss: {test_loss:.4f}")

    # Predictions for detailed results
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test

    fashion_class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Save detailed results
    save_results("fashion_mnist", y_true, y_pred, fashion_class_names, OUTPUT_DIR)

    # -----------------------------
    # 1.5 Save model & plots
    # -----------------------------
    model_path = os.path.join(OUTPUT_DIR, f"fashion_mnist_cnn_{timestamp()}.h5")
    model.save(model_path)
    print(f"Saved Fashion-MNIST CNN model to: {model_path}")

    # Plot training history
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axs[0].plot(history.history["accuracy"], label="train")
    axs[0].plot(history.history["val_accuracy"], label="val")
    axs[0].set_title("Fashion-MNIST Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    # Loss
    axs[1].plot(history.history["loss"], label="train")
    axs[1].plot(history.history["val_loss"], label="val")
    axs[1].set_title("Fashion-MNIST Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"fashion_mnist_history_{timestamp()}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Fashion-MNIST training plots to: {plot_path}")

    return test_acc, model_path, plot_path


# ============================================================
# 2. Pretrained ResNet on CIFAR-100
# ============================================================
def run_cifar100_resnet():
    print("\n=== CIFAR-100: Pretrained ResNet50 ===")

    # -----------------------------
    # 2.1 Load data
    # -----------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
        label_mode="fine"
    )

    num_classes = 100
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Split train into train/validation
    val_fraction = 0.1
    n_val = int(len(x_train) * val_fraction)

    x_val = x_train[:n_val]
    y_val = y_train[:n_val]
    x_train2 = x_train[n_val:]
    y_train2 = y_train[n_val:]

    # -----------------------------
    # 2.2 Preprocessing
    # -----------------------------
    # ResNet50 expects 224x224x3 and special preprocessing
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    # Convert labels to categorical
    y_train2_cat = to_categorical(y_train2, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Data augmentation & resizing/preprocessing
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ],
        name="data_augmentation",
    )

    preprocess_layer = tf.keras.Sequential(
        [
            layers.Resizing(224, 224),
            layers.Lambda(preprocess_input),
        ],
        name="preprocess",
    )

    # -----------------------------
    # 2.3 Build transfer learning model with ResNet50
    # -----------------------------
    inputs = layers.Input(shape=(32, 32, 3))

    x = data_augmentation(inputs)
    x = preprocess_layer(x)

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    # Freeze base model (only train top for now)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_CIFAR100")

    model.summary()

    # -----------------------------
    # 2.4 Compile & train (top layers only)
    # -----------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train2,
        y_train2_cat,
        epochs=10,             # reduce if training is too slow
        batch_size=128,
        validation_data=(x_val, y_val_cat),
        verbose=2,
    )

    # -----------------------------
    # 2.5 Optional fine-tuning:
    #     Unfreeze some layers of the base model
    # -----------------------------
    fine_tune = True
    if fine_tune:
        print("\nUnfreezing last layers of ResNet50 for fine-tuning...")
        for layer in base_model.layers[-50:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_fine = model.fit(
            x_train2,
            y_train2_cat,
            epochs=5,           # small number of fine-tuning epochs
            batch_size=128,
            validation_data=(x_val, y_val_cat),
            verbose=2,
        )

        # Extend training history with fine-tuning
        for k in history.history:
            history.history[k].extend(history_fine.history[k])

    # -----------------------------
    # 2.6 Evaluate
    # -----------------------------
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"CIFAR-100 Test accuracy (ResNet50 TL): {test_acc:.4f}")
    print(f"CIFAR-100 Test loss (ResNet50 TL): {test_loss:.4f}")

    # Predictions for detailed results
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test.flatten()

    # CIFAR-100 fine label names (in official order)
    cifar100_class_names = [
        'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
        'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
        'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
        'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
        'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
        'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
        'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
        'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
        'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
        'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
    ]

    # Save detailed results
    save_results("cifar100_resnet", y_true, y_pred, cifar100_class_names, OUTPUT_DIR)

    # -----------------------------
    # 2.7 Save model & plots
    # -----------------------------
    model_path = os.path.join(OUTPUT_DIR, f"cifar100_resnet50_{timestamp()}.h5")
    model.save(model_path)
    print(f"Saved CIFAR-100 ResNet model to: {model_path}")

    # Plot training history
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axs[0].plot(history.history["accuracy"], label="train")
    axs[0].plot(history.history["val_accuracy"], label="val")
    axs[0].set_title("CIFAR-100 Accuracy (ResNet50)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    # Loss
    axs[1].plot(history.history["loss"], label="train")
    axs[1].plot(history.history["val_loss"], label="val")
    axs[1].set_title("CIFAR-100 Loss (ResNet50)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"cifar100_resnet_history_{timestamp()}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved CIFAR-100 training plots to: {plot_path}")

    return test_acc, model_path, plot_path


# ============================================================
# 3. Main
# ============================================================
if __name__ == "__main__":
    # Custom CNN on Fashion-MNIST
    fashion_acc, fashion_model_path, fashion_plot_path = run_fashion_mnist_cnn()

    # Pretrained ResNet on CIFAR-100
    cifar_acc, cifar_model_path, cifar_plot_path = run_cifar100_resnet()

    print("\n=== Summary ===")
    print(f"Fashion-MNIST CNN test accuracy: {fashion_acc:.4f}")
    print(f"CIFAR-100 ResNet test accuracy: {cifar_acc:.4f}")
    print("All models, plots, and result files are saved in:", OUTPUT_DIR)
