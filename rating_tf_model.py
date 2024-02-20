from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from constants_app import DF_COMMENTAIRE_PLONGEMENT
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from constants import FICHIER_MATRICE_CONFUSION

df = DF_COMMENTAIRE_PLONGEMENT
df = df[df['rating'] != "Null"]
df.loc[:, 'rating'] = df.loc[:, 'rating'].astype(int)
df.loc[:, 'interest'] = df.loc[:, 'rating'] > 5

X = df.loc[:, 'V_000':'V_383'].values
y = df['interest'].replace({False: 0, True: 1}).values

# Diviser les données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, shuffle=True)
# Diviser le jeu de test en jeu de validation et jeu de test final
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=24, shuffle=True)


# Définition du modèle CNN
def create_model():
    model = Sequential([
        Flatten(input_shape=(384,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model


# Création du modèle
model = create_model()

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Définition des callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('data/models/rating_best_model.h5', monitor='val_loss', save_best_only=True,
                                   verbose=1)

# Entraînement du modèle avec les callbacks
history = model.fit(X_train, y_train, epochs=100, batch_size=512,
                    validation_data=(X_val, y_val), verbose=1,
                    callbacks=[early_stopping, model_checkpoint])

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Obtenir les prédictions sous forme de probabilités pour chaque classe
y_pred_probs = model.predict(X_test)

# Convertir les probabilités en classes en prenant la classe avec la probabilité la plus élevée
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

print('Matrice de confusion:')
print(conf_matrix)

# Enregistrer la matrice de confusion
with open(str(FICHIER_MATRICE_CONFUSION), 'wb') as f:
    pickle.dump(conf_matrix, f)

