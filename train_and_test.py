import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.model_selection import train_test_split
from models import model_factory
from utils import normalize_to_range, train_test_split_preserve_distr
from random import choice, sample, randint
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--model_name", type=str, default="InceptionV3", help="Name of model to train")
parser.add_argument("--fold", type=int, default=0, help="Fold number")
parser.add_argument("--gini", type=str, default='0.19', help="Gini number")
parser.add_argument("--upsample", type=str, default='10', help="Upsample number")
parser.add_argument("--name_tag", type=str, default='', help="String to attach at the beginning of the filename")
args = parser.parse_args()

np.random.seed(args.fold)
tf.random.set_seed(args.fold)
initializer = tf.keras.initializers.GlorotUniform(seed=args.fold)

orig_input_shape=(128, 128, 1)
target_input_shape=(96, 96, 1)

model_name = f'{args.model_name}.{args.epochs}'

"""
Augmentation functions
"""
def rotate(image):
    image = image.copy()
    clockwise = choice([0, 1])
    rows, cols = image.shape[:2]
    #angle = np.random.randint(1 + 36*(magnitude-1), 36*magnitude)
    angle = np.random.randint(1, 180)
    angle = angle if clockwise else -angle
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def rotate_four_angles(image):
    image = image.copy()
    rows, cols = image.shape[:2]
    angle = choice([90, 180, 270])
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def vertical_flip(image):
    image = image.copy()
    flip_direction = 0
    return cv2.flip(image, flip_direction)

def horizontal_flip(image):
    image = image.copy()
    flip_direction = 1
    return cv2.flip(image, flip_direction)

def translate(image):
    translation_choice = choice([0, 1, 2])  # 0 for vertical, 1 for horizontal, 2 for both
    magnitude1 = np.random.randint(1, 5)
    #magnitude1 = choice([0, 1])
    magnitude1 = magnitude1 * choice([-1, 1])
    magnitude2 = np.random.randint(1, 5)
    #magnitude2 = choice([0, 1])
    magnitude2 = magnitude2 * choice([-1, 1])
    image = image.copy()
    rows, cols = image.shape[:2]
    if translation_choice == 0:  # Vertical translation
        M = np.float32([[1, 0, 0], [0, 1, magnitude1]])
    elif translation_choice == 1:  # Horizontal translation
        M = np.float32([[1, 0, magnitude2], [0, 1, 0]])
    else:  # Both translations
        M = np.float32([[1, 0, magnitude2], [0, 1, magnitude1]])  # Applying both translations
    return cv2.warpAffine(image, M, (cols, rows))

def random_erasing(image):
    erasing_choice = choice([0, 1, 2])
    magnitude1 = np.random.randint(1, 5)
    magnitude2 = np.random.randint(1, 5)
    if erasing_choice == 0:
        rows, cols = image.shape
        start_col = randint(0, cols - magnitude1)
        image[:, start_col:start_col + magnitude1] = np.random.randint(256, size=(rows, magnitude1))
        return image
    elif erasing_choice == 1:
        rows, cols = image.shape
        start_row = randint(0, rows - magnitude2)
        image[start_row:start_row + magnitude2, :] = np.random.randint(256, size=(magnitude2, cols))
        return image
    else:
        rows, cols = image.shape
        start_col = randint(0, cols - magnitude1)
        image[:, start_col:start_col + magnitude1] = np.random.randint(256, size=(rows, magnitude1))
        start_row = randint(0, rows - magnitude2)
        image[start_row:start_row + magnitude2, :] = np.random.randint(256, size=(magnitude2, cols))
        return image

augment_funcs = {
#    'rotate': rotate,
    'rotate_four_angles': rotate_four_angles,
    'translate': translate,
    'vertical_flip': vertical_flip,
    'horizontal_flip': horizontal_flip,
    'random_erasing': random_erasing,
    'identity': lambda x: x,
}


"""
Data pipeline
"""
# load .csv with information on file names
train_df = pd.read_csv(f'data/ibtracs/ibtracs_gridsat_train.ss_bins.gini_0.10.max_freq_20000.max_upsample_50.csv')
#train_df = pd.read_csv(f'data/ibtracs/ibtracs_gridsat_train2.csv')
test_df = pd.read_csv('data/ibtracs/ibtracs_gridsat_test2.csv')

# take 80% of data as train, 20% as test
ids = list(train_df.IDX)
ids_train_split, ids_valid_split = train_test_split(ids, test_size=0.2, random_state=42)

# Function to rotate and crop the array
def normalize_to_range(arr, lower_bound=0, upper_bound=255):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    # Normalize to [0, 1]
    normalized = (arr - arr_min) / (arr_max - arr_min + 1e-7)
    # Scale to [lower_bound, upper_bound]
    scaled = normalized * (upper_bound - lower_bound) + lower_bound
    return scaled

def crop(array, crop=target_input_shape[0]):
    center = np.array(array.shape) / 2
    x1, y1 = center - crop//2
    x2, y2 = center + crop//2
    cropped_array = array[int(x1):int(x2), int(y1):int(y2)]
    return cropped_array

def load_nc_augment(idx_true, aug_flag=None):
    N = np.random.randint(1, len(list(augment_funcs.keys())))
    funcs = sample(list(augment_funcs.keys()), N)
    with xr.open_dataset(f'data/gridsat_cropped/GRIDSAT.{idx_true}.nc') as data:
        irwin = data['irwin_cdr'].values[0]
        irwin = normalize_to_range(irwin)
        augmented = irwin.copy()
        if aug_flag:
            for func_name in funcs:
                augment_func = augment_funcs[func_name]
                augmented = augment_func(augmented)
        augmented = crop(augmented)
        augmented = np.expand_dims(augmented, axis=-1)
        #augmented = np.repeat(np.expand_dims(augmented, axis=-1), 3, axis=-1)
    return augmented

def data_generator(ids):
    while True:
        for start in range(0, len(ids), args.batch_size):
            x_batch = []
            y_batch = []
            end = min(start + args.batch_size, len(ids))
            ids_batch = ids[start:end]
            for idx in ids_batch:
                aug_flag = train_df.loc[train_df['IDX'] == idx].AUG.values[0]
                idx_true = train_df.loc[train_df['IDX'] == idx].IDX_TRUE.values[0]
                x = load_nc_augment(idx_true, aug_flag)
                y = train_df.loc[train_df['IDX'] == idx].WMO_WIND.values[0]
                x_batch.append(x)
                y_batch.append(y)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch, y_batch

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=6,
                           min_delta=0.1,
                           verbose=1),
             ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1,
                                patience=3,
                                min_delta=0.1,
                                verbose=1 ),
             ModelCheckpoint(monitor='val_loss',
                             filepath=f'saved_weights/{model_name}.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1)]

"""
Load and train model
"""
# Load model
model = model_factory(model_name=args.model_name, input_shape=target_input_shape)
#model.load_weights(filepath=f'saved_weights/{model_name}.hdf5')
#model = inception()
#model.summary()

# Train model
print(f'Training model {model_name}')
history = model.fit(data_generator(ids=ids_train_split),
                    steps_per_epoch=int(np.ceil(len(ids_train_split) / args.batch_size)),
                    epochs=args.epochs,
                    callbacks=callbacks,
                    validation_data=data_generator(ids_valid_split),
                    validation_steps=int(np.ceil(len(ids_valid_split) / args.batch_size)),
                    verbose=2)

best_score = min(history.history['val_root_mean_squared_error'])
print(f'RMSE for {model_name}: {best_score}')


"""
Training plots
"""
# plot MAE
fig, ax = plt.subplots(1, figsize=(14,7))
plt.plot(history.history['root_mean_squared_error'], linewidth=2.5)
plt.plot(history.history['val_root_mean_squared_error'], linewidth=2.5)
plt.ylabel('MAE [mm/3h]', fontsize=24)
plt.xlabel('epoch', fontsize=24)
plt.legend(['train', 'validation'], loc='upper right', fontsize=20)
plt.grid(color='grey', linewidth=0.45)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.set_xlim(left=0, right=len(history.history['loss'])-1)
plt.savefig(f'METRICS.jpg')

# plot loss
fig, ax = plt.subplots(1, figsize=(14,7))
plt.plot(history.history['loss'], linewidth=2.5)
plt.plot(history.history['val_loss'], linewidth=2.5)
plt.ylabel('loss', fontsize=24)
plt.xlabel('epoch', fontsize=24)
plt.legend(['train', 'validation'], loc='upper right', fontsize=20)
plt.grid(color='grey', linewidth=0.45)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.set_xlim(left=0, right=len(history.history['loss'])-1)
plt.savefig(f'LOSS.jpg')


"""
Testing
"""
def load_data(idx):
    with xr.open_dataset(f'data/gridsat_cropped/GRIDSAT.{idx}.nc') as data:
        irwin = data['irwin_cdr'].data[0]
        irwin = normalize_to_range(irwin)
        irwin = crop(irwin)
        irwin = np.expand_dims(irwin, axis=-1)
        #irwin = np.repeat(np.expand_dims(irwin, axis=-1), 3, axis=-1)
    y = test_df.loc[test_df['IDX_TRUE'] == idx].WMO_WIND.values[0]
    y = np.array(y)
    return irwin, y

def categorize(wind_speed):
    if wind_speed < 34:
        return 'TD'
    elif 34 <= wind_speed <= 63:
        return 'TS'
    elif 64 <= wind_speed <= 82:
        return 'Cat1'
    elif 83 <= wind_speed <= 95:
        return 'Cat2'
    elif 96 <= wind_speed <= 112:
        return 'Cat3'
    elif 113 <= wind_speed <= 135:
        return 'Cat4'
    else: # wind_speed > 135
        return 'Cat5'
    
categories = ['TD', 'TS', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'All']
stratified_errors = {cat:[] for cat in categories}
stratified_rmse = {cat:None for cat in categories}

for idx, row in test_df.iterrows():
    # Load data
    idx = test_df.loc[idx, 'IDX_TRUE']
    x, y = load_data(idx)
    x = np.expand_dims(x, axis=0)
    cat = categorize(y)
    y_pred = model.predict(x, verbose=0)
    squared_error = np.square(y - y_pred)
    stratified_errors['All'].append(squared_error)
    stratified_errors[cat].append(squared_error)

print('Category\tRMSE (knots)')
print('----------------------------')
for key in stratified_rmse:
    stratified_rmse[key] = np.sqrt(np.mean(stratified_errors[key]))
    print(f'{key}\t\t{stratified_rmse[key]:.2f}')

# Convert the stratified_rmse dictionary to a DataFrame for easy CSV writing
new_data = pd.DataFrame([stratified_rmse])
new_data['Model'] = model_name

# Try to read the existing CSV file. If it doesn't exist, use the new data as the only data.
try:
    existing_data = pd.read_csv('rmse_results.csv')
    # Concatenate the existing data with the new data
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
except FileNotFoundError:
    updated_data = new_data

# Save the updated DataFrame to CSV, overwriting the old file
updated_data.to_csv('rmse_results.csv', index=False)
