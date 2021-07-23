import os
import shutil
from tqdm.auto import tqdm

DIR = "\\".join(os.path.realpath(__file__).split("\\")[:-1])
TRAINVALTEST_SPLIT = os.path.join(DIR, "UCF-101", "TrainTest")
if os.path.exists(TRAINVALTEST_SPLIT): # if traintest folder exists in UCF-101 folder
    for split in os.listdir(TRAINVALTEST_SPLIT): # for each split ("Train", "Test")
        folder_pbar = tqdm(os.listdir(os.path.join(TRAINVALTEST_SPLIT, split)), leave=False, position=0)
        for folder in folder_pbar: # each class in dataset
            folder_pbar.set_description(f"{split} -> {folder}")
            folder_path = os.path.join(TRAINVALTEST_SPLIT, split, folder)
            frame_pbar = tqdm(os.listdir(folder_path), leave=False, position=1)
            for frame in frame_pbar: # delete each frame in class for split in dataset
                frame_pbar.set_description("Deleting frames")
                frame_path = os.path.join(folder_path, frame)
                os.remove(frame_path)
    shutil.rmtree(TRAINVALTEST_SPLIT)
    print("Done!\n")
else:
    print("TrainTest folder does not exist!\n")
