import os
import keras
from tensorflow import data as tf_data
from keras import layers
import sys

# Copy data loading, augmentation and make model function
data_dir = os.path.join(sys.argv[1])
print("Found data directory with:", os.listdir(data_dir))

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(data_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
print("Images deleted:", num_skipped)

image_size = (180, 180)
batch_size = 32

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

# 1. Load the pretrained model
pretrained_model = keras.models.load_model("model-1.keras")
print("Successfully loaded pretrained model.")

# 2. Create a new model with the same architecture
image_size = (180, 180)
new_model = make_model(input_shape=image_size + (3,), num_classes=2)

# 3. Selectively transfer weights
conv_count = 0
transferred_count = 0

for old_layer, new_layer in zip(pretrained_model.layers, new_model.layers):
    # Skip layers that don't have weights (like Activation, MaxPooling, Dropout)
    if not old_layer.weights:
        continue
        
    # Check if the layer is a convolutional layer
    if isinstance(new_layer, (layers.Conv2D, layers.SeparableConv2D)):
        conv_count += 1
        # Skip the first two convolutional layers (leave them randomly initialized)
        if conv_count <= 2:
            print(f"Skipping weight transfer for: {new_layer.name} (Re-initialized)")
            continue
            
    # Check if it's the output Dense layer
    if isinstance(new_layer, layers.Dense):
        print(f"Skipping weight transfer for: {new_layer.name} (Re-initialized)")
        continue

    # For all other layers, copy the pre-trained weights
    new_layer.set_weights(old_layer.get_weights())
    transferred_count += 1

print(f"Successfully transferred weights for {transferred_count} layers.")
print("The first 2 Conv layers and the Dense output layer are starting from scratch.")

# 4. Compile the new model
new_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

epochs = 50
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

# 5. Fit the model
new_model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds
)