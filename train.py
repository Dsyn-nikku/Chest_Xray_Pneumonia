
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. CONFIGURATION
# -----------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_PATH = "chest_xray/train"
TEST_PATH = "chest_xray/test"

tf.keras.utils.set_random_seed(42)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

print("Loading training images...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    validation_split=0.2,
    subset="training",
    seed=42
)

print("Loading validation images...")

validation_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    validation_split=0.2,
    subset="validation",
    seed=42
)

print("Loading test images...")

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["NORMAL", "PNEUMONIA"],
    shuffle=False
)

# -----------------------------
# 3. OPTIMIZE DATASET
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(AUTOTUNE)
validation_data = validation_data.cache().prefetch(AUTOTUNE)
test_data = test_data.cache().prefetch(AUTOTUNE)

# -----------------------------
# 4. BUILD CNN MODEL
# -----------------------------

model = keras.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.Rescaling(1./255),

    layers.RandomRotation(0.03),
    layers.RandomZoom(0.08),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")
])

# -----------------------------
# 5. COMPILE MODEL
# -----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# -----------------------------
# 6. TRAIN MODEL
# -----------------------------

os.makedirs("models", exist_ok=True)

callbacks = [

    keras.callbacks.ModelCheckpoint(
        "models/chest_xray_model.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    )
]

print("\nStarting model training...")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# 7. EVALUATE MODEL
# -----------------------------

print("\nEvaluating model on test dataset...")

best_model = keras.models.load_model(
    "models/chest_xray_model.keras"
)

results = best_model.evaluate(test_data, return_dict=True)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# -----------------------------
# 8. SAVE TRAINING GRAPH
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.savefig("models/accuracy_graph.png")
plt.show()

print("\nMODEL TRAINING COMPLETED!")
print("Model saved successfully!")
