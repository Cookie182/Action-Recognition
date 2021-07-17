import os
import shutil
from tqdm.auto import tqdm

DIR = "\\".join(os.path.realpath(__file__).split("\\")[:-1])
TRAINVALTEST_SPLIT = os.path.join(DIR, "UCF-101", "TrainTest")

if os.path.exists(TRAINVALTEST_SPLIT):
    split_pbar = tqdm(os.listdir(TRAINVALTEST_SPLIT), leave=False, colour='white', position=1)
    for split in split_pbar:
        split_pbar.set_description(f"Split - {split}")
        for label_folder in os.listdir(os.path.join(TRAINVALTEST_SPLIT, split)):
            frame_pbar = tqdm(os.listdir(os.path.join(TRAINVALTEST_SPLIT, split, label_folder)), leave=False, colour='white', position=2)
            for frame in frame_pbar:
                frame_pbar.set_description(f"Deleting {frame}")
                os.remove(os.path.join(TRAINVALTEST_SPLIT, split, label_folder, frame))
    shutil.rmtree(TRAINVALTEST_SPLIT)
    print("Done!\n")
else:
    print("TrainTest folder does not exist!\n")
