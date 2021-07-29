import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # nopep8
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
import argparse

VERBOSE = 1


def evaluatemodel(model, filepath, modelname, modelpath, train_gen, val_gen, test_gen, batchsize, epochs):
    """train and evaluate model then display the results, also save, optionally, performance metrics graphs and model structure

    Args:
        model (tensorflow model): tensorflow model to train/evaluate/test
        filepath (str): absolute path of model script files
        modelname (str): name of the model file
        modelpath (str): absolute path to store the trained model and store performance graphs and
        train_gen (tensorflow data generator): training data generator
        val_gen (tensorflow data generator): validation data generator
        test_gen (tensorflow data generator): test data generator
        batchsize (int): batch size of training/validation data
        epochs (int): number of epochs to train model
    """
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=VERBOSE)
    callbacks = [earlystopping]

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help='save the model', action='store_true')
    args = parser.parse_args()
    if args.save:
        model_png_path = os.path.join(*filepath.split("\\")[3:-1], f"{modelname}.png")
        keras.utils.plot_model(model, to_file=model_png_path, show_shapes=True)
        best_checkpoint = keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                          monitor='val_acc',
                                                          save_best_only=True,
                                                          save_freq='epoch',
                                                          verbose=VERBOSE)
        callbacks.append(best_checkpoint)
        print("\nModel IS being saved after every epoch!\n")
    else:
        print("\nModel is NOT being saved!\n")

    train = model.fit(train_gen,
                      epochs=epochs,
                      verbose=VERBOSE,
                      steps_per_epoch=len(train_gen) // batchsize,
                      callbacks=callbacks,
                      validation_data=val_gen,
                      validation_steps=len(val_gen) // batchsize,
                      use_multiprocessing=True,
                      workers=-1)

    test = model.evaluate(test_gen, steps=len(test_gen) // batchsize, workers=-1, use_multiprocessing=True, verbose=VERBOSE)

    train_history = pd.DataFrame(train.history)
    plt.figure(figsize=(8, 6))
    plt.title(f"Test stats:\nLoss: {test[0]} \nAcc: {test[1]}")
    for label in train_history.keys():
        plt.plot(train_history[label], label=label, linestyle='--' if label[:3] == 'val' else '-')
    plt.xlabel('Epochs')
    plt.legend()
    plt.margins(x=0, y=0)
    plt.tight_layout()
    if args.save:
        plt.savefig(os.path.join(*filepath.split("\\")[2:-1], f"{modelname}_performance.png"))
    plt.show()

    return "\nThank you for using the script!"
