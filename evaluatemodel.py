import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # nopep8
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt


def evaluatemodel(model, filepath, modelname, train_gen, val_gen, test_gen, batchsize, epochs, patience=3, verbose=1, save=False):
    """train and evaluate model then display the results, also save, optionally, performance metrics graphs and model structure

    Args:
        model (tensorflow model): tensorflow model to train/evaluate/test
        filepath (str): absolute path of model script files
        modelname (str): name of the model file
        train_gen (tensorflow data generator): training data generator
        val_gen (tensorflow data generator): validation data generator
        test_gen (tensorflow data generator): test data generator
        batchsize (int): batch size of training/validation data
        epochs (int): number of epochs to train model
        patience (int): number of epochs to wait and see if monitored value improves during training. Defaults to 3.
        verbose (int): verbosity levels for certain functions. Defaults to 1.
        save (bool): whether to save the model, and its training performance metrics. Defaults to False.
    """
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
    callbacks = [earlystopping]

    dir_path = filepath.split("\\")[-2]
    if save:
        model_png_path = os.path.join(dir_path, f"{modelname}_structure.png")
        keras.utils.plot_model(model, to_file=model_png_path, show_shapes=True)

        model_save_path = os.path.join(dir_path, f"{modelname}.h5")
        best_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                          monitor='val_acc',
                                                          save_best_only=True,
                                                          save_freq='epoch',
                                                          verbose=verbose)
        callbacks.append(best_checkpoint)
        print("\nModel IS being saved after every epoch!\n")
    else:
        print("\nModel is NOT being saved!\n")

    print("Training model: ")
    train = model.fit(train_gen,
                      epochs=epochs,
                      verbose=verbose,
                      steps_per_epoch=len(train_gen) // batchsize,
                      callbacks=callbacks,
                      validation_data=val_gen,
                      validation_steps=len(val_gen) // batchsize,
                      use_multiprocessing=True,
                      workers=-1)

    print("\nEvaluating Model: ")
    test = model.evaluate(test_gen,
                          steps=len(test_gen) // batchsize,
                          workers=-1, use_multiprocessing=True, verbose=verbose)

    train_history = pd.DataFrame(train.history)

    plt.figure(figsize=(8, 6))
    plt.title(f"Test stats:\nLoss: {round(test[0], 4)} \nAcc: {round(test[1], 4)}")
    for label in train_history.keys():
        plt.plot(train_history[label], label=label, linestyle='--' if label[:3] == 'val' else '-')
    plt.xlabel('Epochs')
    plt.legend()
    plt.margins(x=0, y=0)
    plt.tight_layout()
    if save:
        for column in train_history.columns:
            train_history[column] = [round(x, 5) for x in train_history[column]]
        train_history.insert(0, 'Epochs', range(1, len(train_history) + 1))
        plt.savefig(os.path.join(dir_path, f"{modelname}_training_performance.png"))
        train_history.to_csv(os.path.join(dir_path, f"{modelname}_training_performance.txt"),
                             header=train_history.columns, index=None)

    plt.show()
    print("\nThank you for using the script!")
