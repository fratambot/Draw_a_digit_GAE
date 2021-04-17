import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# -- Utility functions


# plot performances
def model_performance(histories):
    fig = plt.figure(figsize=(15, 5))
    for i in range(len(histories)):
        # plot loss
        plt.subplot(1, 2, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue')
        plt.plot(histories[i].history['val_loss'], color='orange')
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.legend(['train', 'test'], loc='upper right')
        # plot accuracy
        plt.subplot(1, 2, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue')
        plt.plot(histories[i].history['val_accuracy'], color='orange')
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    fig.savefig('model_performance_crossval.png')


# score
def score(model, testX, testY):
    sco = model.evaluate(
        testX,
        to_categorical(testY)
    )
    result = 'This model achieved {:.3f} test loss and {:.2f} % test accuracy'.format(
        sco[0], sco[1]*100)
    return result


# Dataset loading and pre-treatment
print('loading dataset...')
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = (trainX / 255) - 0.5
testX = (testX / 255) - 0.5
trainX = np.expand_dims(trainX, axis=3)
testX = np.expand_dims(testX, axis=3)


# define CNN model
def define_model():
    print('defining the model...')
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


print('model created:')
print(define_model().summary())

# save the model summary
with open('model_summary.txt', 'w') as fh:
    define_model().summary(print_fn=lambda x: fh.write(x + '\n'))


# evaluate model with k-fold cross-validation
def evaluate_model(model, dataX, dataY, n_folds=5, epochs=10, verbose=0, batch_size=32):
    print('evaluating the model...')
    histories = list()
    kfold = KFold(n_folds, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(dataX):
        model = define_model()
        trainX, testX = dataX[train_index], dataX[test_index]
        trainY, testY = dataY[train_index], dataY[test_index]
        # fit model
        history = model.fit(
            trainX,
            to_categorical(trainY),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(testX, to_categorical(testY)),
            verbose=verbose
        )
        histories.append(history)
    return histories

# Uncomment to perform cross-validation and produce a plot on perf

# histories = evaluate_model(trainX, trainY)
# model_performance(histories)


model = define_model()

print('fitting the model...')
model.fit(
    trainX,
    to_categorical(trainY),
    epochs=10,
    batch_size=32
)

# score
score(model, testX, testY)

# save model
model.save('32C5-P2_64C5-P2-128.h5')
