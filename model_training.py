import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


DATA_DIR = 'datasets/labeled'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


IMG_SIZE = (48,48)
BATCH = 64
EPOCHS = 30


train_datagen = ImageDataGenerator(rescale=1./255,
validation_split=0.2,
rotation_range=10,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.1,
horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
DATA_DIR,
target_size=IMG_SIZE,
color_mode='grayscale',
batch_size=BATCH,
class_mode='categorical',
subset='training')


val_generator = train_datagen.flow_from_directory(
DATA_DIR,
target_size=IMG_SIZE,
color_mode='grayscale',
batch_size=BATCH,
class_mode='categorical',
subset='validation')


num_classes = train_generator.num_classes


model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
BatchNormalization(),
MaxPooling2D((2,2)),


Conv2D(64, (3,3), activation='relu'),
BatchNormalization(),
MaxPooling2D((2,2)),


Conv2D(128, (3,3), activation='relu'),
BatchNormalization(),
MaxPooling2D((2,2)),


Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(num_classes, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)


model_path = os.path.join(MODEL_DIR, 'emotion_model.h5')
model.save(model_path)
print('Saved model to', model_path)