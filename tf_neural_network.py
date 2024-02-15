import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers.legacy import Adam
# from tensorflow.keras.regularizers import l2
import tensorflow.keras.callbacks as callbacks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import datetime
import locale
from pathlib import Path

from constants import FICHIER_COMMENTAIRE_PLONGEMENT, DOSSIER_MODELS

if __name__ == "__main__":
    print("start")
    print("loading data ...")

    df = pd.read_csv(FICHIER_COMMENTAIRE_PLONGEMENT)

    data_length = len(set(df['titre'].values))
    X = df.iloc[:, 4:].values
    y = df.iloc[:, 0].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("train test split ....")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, shuffle=True, random_state=33)
    del X, y, y_encoded, df

    INPUT_SHAPE = X_train.shape[1]
    print("INPUT_SHAPE : ", INPUT_SHAPE)
    OUTPUT_SHAPE = data_length
    print("OUTPUT_SHAPE : ", OUTPUT_SHAPE)

    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    epochs = 100
    batch_size = 512
    lr = 0.0001

    model_name = str(Path(DOSSIER_MODELS) / ("model_" + str(INPUT_SHAPE) + "_batch_" + str(batch_size) +
                                             "_lr_" + str(lr).split(".")[1] + "_" + current_time_str + ".keras"))
    checkpoint = callbacks.ModelCheckpoint(
        model_name,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        verbose=1,
        mode='max',
        restore_best_weights=True
    )

    model = Sequential()

    model.add(BatchNormalization(input_shape=(INPUT_SHAPE,)))
    model.add(Dense(768, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Dense(384, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(BatchNormalization())
    model.add(Dense(OUTPUT_SHAPE, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    model.summary()

    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Compiler le modèle
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weight_dict,
    )

    # Évaluer le modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Perte sur les données d'entraînement
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Précision sur les données d'entraînement
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
