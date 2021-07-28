## Action-Recognition
**University Summer Internship**
Comparing my model, [first_model](https://github.com/Cookie182/Action-Recognition/tree/main/first_model) with the performance of [VGG16](https://github.com/Cookie182/Action-Recognition/tree/main/Custom%20VGG16) and [VGG19](https://github.com/Cookie182/Action-Recognition/tree/main/Custom%20VGG19), currently. More models will be created and first_model will be updated as more models are created to compare performance metrics with. 

[trainvaltest.py](https://github.com/Cookie182/Action-Recognition/blob/main/trainvaltest.py): A script to navigate to the UCF-101 dataset and extract every nth frame from a video and store it in a **Train** and **Test** folder to be used for the aforementioned purposes with a model. This script also prepares training, validation and test data generators from the directory to be used when training and evaluating models. All scripts and notebooks of models call upon this script initially. 

[trainvaltest_delete.py](https://github.com/Cookie182/Action-Recognition/blob/main/trainvaltest_delete.py): Script to navigate to the TrainTest folder in the UCF-101 dataset created when the trainvaltest.py script is ran, and delete it's contents and the TrainTest folder.

Link to [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)


## **DIRECTORY STRUCUTRE**
.
|-- Custom VGG16
|   |-- my_VGG16.h5
|   |-- my_VGG16.ipynb
|   |-- my_VGG16.png
|   `-- my_vgg16.py
|-- Custom VGG19
|   |-- my_VGG19.h5
|   |-- my_VGG19.h5.png
|   |-- my_VGG19.ipynb
|   `-- my_vgg19.py
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
|   |-- Preprocessing.cpython-39.pyc
|   |-- PreprocessingLayers.cpython-39.pyc
|   |-- preprocess.cpython-39.pyc
|   `-- trainvaltest.cpython-39.pyc
|-- directory.txt
|-- first_model
|   |-- model1.py
|   |-- model1_notebook.ipynb
|   |-- the_first_war.h5
|   `-- the_first_war.png
|-- preprocess.py
|-- testing
|   `-- testing.py
|-- trainvaltest.py
`-- trainvaltest_delete.py

108 directories, 22 files
