## Action-Recognition
***University Summer Internship***
Comparing my model, [My Model](https://github.com/Cookie182/Action-Recognition/tree/main/My%20Model) with the performance of [VGG16](https://github.com/Cookie182/Action-Recognition/tree/main/Custom%20VGG16) and [VGG19](https://github.com/Cookie182/Action-Recognition/tree/main/Custom%20VGG19), currently. More models may be added later on and thus, based on their performance, My Model may be updated later on.

[preprocess.py](https://github.com/Cookie182/Action-Recognition/blob/main/preprocess.py): A script that stores the preprocessing layers that all the models will use as the first layer. Contains layers that:
  * Rescales the values
  * Applies random rotation to data
  * Applies random zoom to data
  * Applies random translation to data
  * Applies random contrast to data
  * Applies random (horizontal) flipping to data

[trainvaltest.py](https://github.com/Cookie182/Action-Recognition/blob/main/trainvaltest.py): A script to navigate to the UCF-101 dataset and extract every nth frame from a video and store it in a **Train** and **Test** folder to be used for the aforementioned purposes with a model. This script also prepares training, validation and test data generators from the directory to be used when training and evaluating models. All scripts of models call upon this script initially. 

[trainvaltest_delete.py](https://github.com/Cookie182/Action-Recognition/blob/main/trainvaltest_delete.py): Script to navigate to the TrainTest folder in the UCF-101 dataset created when the trainvaltest.py script is ran, and delete it's contents and the TrainTest folder.

[evaluatemodel.py](https://github.com/Cookie182/Action-Recognition/blob/main/evaluatemodel.py): Script that trains and evaluates a model, gives an option (use '--save' when running a model script) to save the trained model, to save png image of the model performance, model structure and a .csv file that documents the accuracy and loss for training and validation sets when the model is being trained.

Link to [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

**EACH MODEL RUNS FOR ENOUGH EPOCHS TO LET THE MODEL RUN FOR APPROX. 10 HOURS UNLESS EARLYSTOPPING KICKS IN**

*SEED VALUE TO BE USED IN ALL APPLICABLE INSTANCES: 182*

*EACH MODEL SCRIPT WHEN RUNNING HAS TWO ARGUEMENTS THAT CAN BE PASSED VIA COMMAND LINE: --save and --valistest*
* *--save: to save the trained model in the same directory of model file, .png image of model training performance, .csv file that documents all metrics when training, and .png image of model structure*
* *--valistest: to use validation data (20%) from the test dataset rather than, by default, the train dataset*

## **DIRECTORY STRUCUTRE**
```BASH
.
|-- Custom VGG16
|   |-- README.md
|   |-- my_VGG16.h5
|   |-- my_VGG16_structure.png
|   |-- my_VGG16_training_performance.png
|   |-- my_VGG16_training_performance.txt
|   `-- my_vgg16.py
|-- Custom VGG19
|   |-- README.md
|   |-- my_VGG19.h5
|   |-- my_VGG19_structure.png
|   |-- my_VGG19_training_performance.png
|   |-- my_VGG19_training_performance.txt
|   `-- my_vgg19.py
|-- My Model
|   |-- README.md
|   |-- my_model.py
|   |-- the_first_war.h5
|   |-- the_first_war_structure.png
|   |-- the_first_war_training_performance.png
|   `-- the_first_war_training_performance.txt
|-- README.md
|-- UCF-101
|   |-- ApplyEyeMakeup
|   |-- ApplyLipstick
|   |-- Archery
|   |-- BabyCrawling
|   |-- BalanceBeam
|   |-- BandMarching
|   |-- BaseballPitch
|   |-- Basketball
|   |-- BasketballDunk
|   |-- BenchPress
|   |-- Biking
|   |-- Billiards
|   |-- BlowDryHair
|   |-- BlowingCandles
|   |-- BodyWeightSquats
|   |-- Bowling
|   |-- BoxingPunchingBag
|   |-- BoxingSpeedBag
|   |-- BreastStroke
|   |-- BrushingTeeth
|   |-- CleanAndJerk
|   |-- CliffDiving
|   |-- CricketBowling
|   |-- CricketShot
|   |-- CuttingInKitchen
|   |-- Diving
|   |-- Drumming
|   |-- Fencing
|   |-- FieldHockeyPenalty
|   |-- FloorGymnastics
|   |-- FrisbeeCatch
|   |-- FrontCrawl
|   |-- GolfSwing
|   |-- Haircut
|   |-- HammerThrow
|   |-- Hammering
|   |-- HandstandPushups
|   |-- HandstandWalking
|   |-- HeadMassage
|   |-- HighJump
|   |-- HorseRace
|   |-- HorseRiding
|   |-- HulaHoop
|   |-- IceDancing
|   |-- JavelinThrow
|   |-- JugglingBalls
|   |-- JumpRope
|   |-- JumpingJack
|   |-- Kayaking
|   |-- Knitting
|   |-- LongJump
|   |-- Lunges
|   |-- MilitaryParade
|   |-- Mixing
|   |-- MoppingFloor
|   |-- Nunchucks
|   |-- ParallelBars
|   |-- PizzaTossing
|   |-- PlayingCello
|   |-- PlayingDaf
|   |-- PlayingDhol
|   |-- PlayingFlute
|   |-- PlayingGuitar
|   |-- PlayingPiano
|   |-- PlayingSitar
|   |-- PlayingTabla
|   |-- PlayingViolin
|   |-- PoleVault
|   |-- PommelHorse
|   |-- PullUps
|   |-- Punch
|   |-- PushUps
|   |-- Rafting
|   |-- RockClimbingIndoor
|   |-- RopeClimbing
|   |-- Rowing
|   |-- SalsaSpin
|   |-- ShavingBeard
|   |-- Shotput
|   |-- SkateBoarding
|   |-- Skiing
|   |-- Skijet
|   |-- SkyDiving
|   |-- SoccerJuggling
|   |-- SoccerPenalty
|   |-- StillRings
|   |-- SumoWrestling
|   |-- Surfing
|   |-- Swing
|   |-- TableTennisShot
|   |-- TaiChi
|   |-- TennisSwing
|   |-- ThrowDiscus
|   |-- TrainTest
|   |-- TrampolineJumping
|   |-- Typing
|   |-- UnevenBars
|   |-- VolleyballSpiking
|   |-- WalkingWithDog
|   |-- WallPushups
|   |-- WritingOnBoard
|   `-- YoYo
|-- __pycache__
|   |-- evaluatemodel.cpython-39.pyc
|   |-- preprocess.cpython-39.pyc
|   `-- trainvaltest.cpython-39.pyc
|-- directory.txt
|-- evaluatemodel.py
|-- preprocess.py
|-- testing
|   `-- testing.py
|-- trainvaltest.py
`-- trainvaltest_delete.py

108 directories, 28 files
```
