## Action-Recognition
**University Summer Internship**
Comparing my model, [first_model](https://github.com/Cookie182/Action-Recognition/tree/main/first_model) with the performance of [VGG16](https://github.com/Cookie182/Action-Recognition/tree/main/Custom%20VGG16) and [VGG19](https://github.com/Cookie182/Action-Recognition/tree/main/Custom%20VGG19), currently. More models will be created and first_model will be updated as more models are created to compare performance metrics with. 

[trainvaltest.py](https://github.com/Cookie182/Action-Recognition/blob/main/trainvaltest.py): A script to navigate to the UCF-101 dataset and extract every nth frame from a video and store it in a **Train** and **Test** folder to be used for the aforementioned purposes with a model. This script also prepares training, validation and test data generators from the directory to be used when training and evaluating models. All scripts and notebooks of models call upon this script initially. 

[trainvaltest_delete.py](https://github.com/Cookie182/Action-Recognition/blob/main/trainvaltest_delete.py): Script to navigate to the TrainTest folder in the UCF-101 dataset created when the trainvaltest.py script is ran, and delete it's contents and the TrainTest folder.

Link to [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)