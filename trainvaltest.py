import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # nopep8

FILE_PATH = '\\'.join(os.path.realpath(__file__).split('\\')[:3])  # path to directory for project
from tqdm.auto import tqdm
import pandas as pd
import cv2
import random
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH = os.path.join(FILE_PATH, "UCF-101")  # path for dataset
LABELS = tuple([x for x in os.listdir(PATH) if x not in ['TrainTest']])
IMG_SIZE = (224, 224)
N_FRAMES = 5  # only save every nth frame
SEED = 123
print("Amount of labels ->", len(LABELS), '\n')

sample_sizes = []
for label in LABELS:
    label_path = os.path.join(PATH, label)
    sample_sizes.append((label, len(os.listdir(label_path))))

max_sample_size = max(sample_sizes, key=lambda i: i[1])
min_sample_size = min(sample_sizes, key=lambda i: i[1])
print(f"{max_sample_size[0]} has the most samples with {max_sample_size[1]} clips")
print(f"{min_sample_size[0]} has the least samples with {min_sample_size[1]} clips\n")

samples_count = []
for label in LABELS:
    sample_count = 1
    label_path = os.path.join(PATH, label)
    clips = os.listdir(label_path)
    while True:
        sample_name = f"g0{sample_count}" if sample_count <= 9 else f"g{sample_count}"

        if len([x for x in clips if sample_name in x]) > 0:
            sample_count += 1
        else:
            samples_count.append((label, sample_count - 1))
            break

print("All actions have:", *set([x[1] for x in samples_count]), "samples\n")

sample_sizes_df = pd.DataFrame(zip([x[1] for x in sample_sizes],
                                   [x[1] for x in samples_count],
                                   LABELS,
                                   [round(x / y, 2) for x, y in zip([x[1] for x in sample_sizes],
                                                                    [y[1] for y in samples_count])]),
                               columns=["Total Clips", "Samples", "Labels", "Clips per sample"]).set_index("Labels")

print("Top 5 actions with least clips per sample:\n", sample_sizes_df.sort_values('Clips per sample', ascending=True).head(), '\n')
print("Top 5 actions with most clips per sample\n", sample_sizes_df.sort_values('Clips per sample', ascending=False).head())

label_trainvaltest_split = dict()

for label in LABELS:
    label_path = os.path.join(PATH, label)
    clip_names = tuple(os.listdir(label_path))

    samples = []
    for i in range(1, 26):  # since each label has 25 samples
        clip_name = f"g0{i}" if i < 10 else f"g{i}"

        samples.append(tuple([clip for clip in clip_names if clip_name in clip]))  # verify if each label has 25 different videos with samples

    Test = tuple(random.choices(samples, k=7))
    Train = tuple([x for x in samples if x not in Test])

    label_trainvaltest_split[label] = {
        'Train': Train,
        'Test': Test
    }

split_path = os.path.join(PATH, "TrainTest")
train_path = os.path.join(split_path, "Train")
test_path = os.path.join(split_path, "Test")

if not os.path.exists(split_path):
    print('\nGenerating train/validation/test data\n')
    os.makedirs(train_path)
    os.makedirs(test_path)

    label_tqdm = tqdm(LABELS, position=0, leave=False, colour='white')
    for label in label_tqdm:
        label_tqdm.set_description(label)
        for split in label_trainvaltest_split[label].keys():
            train_split_path = os.path.join(split_path, split, label)
            if not os.path.exists(train_split_path):
                os.makedirs(train_split_path)

            sample_count = 1
            frame_count = 1
            for sample in label_trainvaltest_split[label][split]:
                sample_tqdm = tqdm(sample, position=1, leave=False, colour='white')
                for clip in sample_tqdm:
                    sample_tqdm_desc = f"{split} -> {sample_count}/{len(label_trainvaltest_split[label][split])}"
                    sample_tqdm.set_description(sample_tqdm_desc)
                    clip = cv2.VideoCapture(os.path.join(PATH, label, clip))
                    while clip.isOpened():
                        success, frame = clip.read()

                        if not success:
                            break

                        if frame_count % N_FRAMES == 0:
                            frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
                            frame_path = os.path.join(train_split_path, f"{label}_{frame_count//N_FRAMES}.jpeg")
                            cv2.imwrite(frame_path, frame)

                        frame_count += 1
                    clip.release()
                sample_count += 1
    print("\nFinished generating train/test data\n")
else:
    print("\ntrain/test data already created\n")

parser = argparse.ArgumentParser()
parser.add_argument('--valistest', help='use part of test data as validation data instead from training data', action='store_true')
parser.add_argument('--save', help='save the model and training performance', action='store_true')
args = parser.parse_args()


def trainvaltest(train_path=train_path, test_path=test_path, LABELS=LABELS, BATCH_SIZE=8, validation_split=0.2, D_TYPE=tf.float64, valistest=args.valistest, save=args.save):
    """Split data into 3 splits (train/test/split) and return after split into batches and

    Args:
        train_path (direcotry): directory to folder of training data
        test_path (directory): directory to folder of testing data
        LABELS (str, optional): labels of dataset. Defaults to LABELS.
        BATCH_SIZE (int, optional): size of batches for splits. Defaults to 8.
        validation_split (float, optional): percentage of data to use as validation set. Defaults to 0.2.
        D_TYPE (data type, optional): data type of data to be used. Defaults to tf.float64
        valistest (bool): command line argument. whether validation data is from test/training set. Defaults to False.
        save (bool): whether to save the model, and its training performance metrics. Defaults to False.

    Returns:
        number of labels, input shape, train/val/test generators, save variable to be used in evaluatemodel.py script
    """
    train_datagen = ImageDataGenerator(data_format='channels_last',
                                       validation_split=validation_split if not valistest else 0,
                                       dtype=D_TYPE)

    test_datagen = ImageDataGenerator(data_format='channels_last',
                                      validation_split=0 if not valistest else validation_split,
                                      dtype=D_TYPE)

    print("Creating training data generator...")
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        batch_size=BATCH_SIZE,
                                                        color_mode='rgb',
                                                        class_mode='sparse',
                                                        subset='training',
                                                        shuffle=True,
                                                        target_size=IMG_SIZE,
                                                        seed=SEED)

    print(f"\nCreating validation data (from {'test' if valistest else 'train'} dataset) generator...")
    val_generator = test_datagen.flow_from_directory(test_path,
                                                     batch_size=BATCH_SIZE,
                                                     color_mode='rgb',
                                                     class_mode='sparse',
                                                     subset='validation',
                                                     shuffle=True,
                                                     target_size=IMG_SIZE,
                                                     seed=SEED) if valistest else train_datagen.flow_from_directory(train_path,
                                                                                                                    batch_size=BATCH_SIZE,
                                                                                                                    color_mode='rgb',
                                                                                                                    class_mode='sparse',
                                                                                                                    subset='validation',
                                                                                                                    shuffle=True,
                                                                                                                    target_size=IMG_SIZE,
                                                                                                                    seed=SEED)

    print("\nCreating testing data generator...")
    test_generator = test_datagen.flow_from_directory(test_path,
                                                      batch_size=BATCH_SIZE,
                                                      color_mode='rgb',
                                                      class_mode='sparse',
                                                      subset='training',
                                                      shuffle=True,
                                                      target_size=IMG_SIZE,
                                                      seed=SEED)

    INPUT_SHAPE = (*IMG_SIZE, 3)
    print(f"\nInput shape -> {INPUT_SHAPE}\n")

    return len(LABELS), INPUT_SHAPE, train_generator, val_generator, test_generator, save
