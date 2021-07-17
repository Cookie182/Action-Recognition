import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # nopep8

FILE_PATH = '\\'.join(os.path.realpath(__file__).split('\\')[:3])  # path to directory for project
from tqdm.auto import tqdm
import pandas as pd
import cv2
import random

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
                                   [round(x / y, 2) for x, y in zip([x[1] for x in sample_sizes], [x[1] for x in samples_count])]),
                               columns=["Total Clips", "Samples", "Labels", "Clips per sample"]).set_index("Labels")
print("Top 5 actions with least clips per sample:\n", sample_sizes_df.sort_values('Clips per sample', ascending=True).head(), '\n')
print("Top 5 actions with most clips per sample\n", sample_sizes_df.sort_values('Clips per sample', ascending=False).head())

label_trainvaltest_split = dict()

for label in LABELS:
    label_path = os.path.join(PATH, label)
    clip_names = tuple([sample for sample in os.listdir(label_path)])

    samples = []
    for i in range(1, 26):  # since each label has 25 samples
        clip_name = f"g0{i}" if i < 10 else f"g{i}"

        samples.append([clip for clip in clip_names if clip_name in clip])

    Test = random.choices(samples, k=6)
    Train = [x for x in samples if x not in Test]

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
    print("\nFinished generating train/validation/test data\n")


def trainvaltest(LABELS=LABELS, BATCH_SIZE=8):
    """Split data into 3 splits (train/test/split) and return after split into batches and

    Args:
        LABELS (str, optional): labels of dataset. Defaults to LABELS.
        BATCH_SIZE (int, optional): size of batches for splits. Defaults to 8.
        show_image (bool, optional): show an example image and its label. Defaults to False.

    Returns:
        tuples: returns the amount of labels, input shape for input layer and generator for each dataset split (train, validation, test)
    """

    D_TYPE = tf.float32
    print("\nTrain:")
    train_datagen = ImageDataGenerator(data_format='channels_last',
                                       validation_split=0.1,
                                       dtype=D_TYPE)

    train_generator = train_datagen.flow_from_directory(train_path,
                                                        batch_size=BATCH_SIZE,
                                                        color_mode='rgb',
                                                        class_mode='sparse',
                                                        shuffle=True,
                                                        target_size=IMG_SIZE,
                                                        seed=SEED)
    print("Validation:")
    validation_generator = train_datagen.flow_from_directory(train_path,
                                                             batch_size=BATCH_SIZE,
                                                             color_mode='rgb',
                                                             class_mode='sparse',
                                                             subset='validation',
                                                             shuffle=True,
                                                             target_size=IMG_SIZE,
                                                             seed=SEED)

    print("Test:")
    test_datagen = ImageDataGenerator(data_format='channels_last',
                                      dtype=D_TYPE)

    test_generator = test_datagen.flow_from_directory(test_path,
                                                      batch_size=BATCH_SIZE,
                                                      color_mode='rgb',
                                                      class_mode='sparse',
                                                      shuffle=True,
                                                      target_size=IMG_SIZE,
                                                      seed=SEED)

    INPUT_SHAPE = None
    for image, label in train_generator:
        if INPUT_SHAPE == None:
            INPUT_SHAPE = image[0].shape
        print("\nInput shape ->", INPUT_SHAPE, '\n')
        break

    return len(LABELS), INPUT_SHAPE, train_generator, validation_generator, test_generator
