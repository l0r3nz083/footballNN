import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import itertools
import shutil

def create_network(input_dim, neurons_1layer, neurons_2layer, activation_function):
    inputs = tf.keras.Input((input_dim,))
    x = layers.Dense(neurons_1layer, activation_function)(inputs)
    x = layers.Dense(neurons_2layer, activation_function)(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(3, "softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="neural_net")
    return model

best_params = {'learning_rate': 0.001, 'epochs': 5, 'neurons_1layer': 50, 'neurons_2layer': 30, 'activation_function': 'tanh', 'batch_size': 100}

tr = pd.read_csv("matches_final.csv", index_col=False)
columns_to_drop = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot',
                   'goal_diff', 'day', 'pk', 'pkatt', 'fk',
                   'referee', 'dist','points', 'hour', 'result_encoded', 'day_code']
tr = tr.drop(columns=columns_to_drop)
num_cols = tr.select_dtypes(include=np.number).columns
num_cols = num_cols.drop(['season'])
num_cols = num_cols.tolist()
cat_cols = tr.select_dtypes(exclude=np.number).columns
cat_cols = cat_cols.drop(['result', 'date'])

tr.dropna(inplace=True)

tr.columns = tr.columns.str.strip()
tr = tr[tr.columns.tolist()[1:]]

tr['time'] = tr['time'].astype('category')
value_counts = tr.time.value_counts()
to_replace = value_counts[value_counts < 102].index
tr['time'] = tr['time'].replace(to_replace, 'Altro')

cat_cols = tr.select_dtypes(exclude=np.number).columns.tolist()
num_cols = tr.select_dtypes(include=np.number).columns.tolist()
predictors = num_cols + cat_cols

# Stacco l'attributo su cui voglio prevedere
X = tr.drop('result', axis=1)
y = tr['result']

# Separo le variabili numeriche da quelle categoriche, perchè le categoriche devo trasformarle
# applicando l'Hot Encoding, mentre le variabili numeriche vengono standardizzate

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# One-Hot Encoding per le variabili categoriche
X_categorical_encoded = pd.get_dummies(X[categorical_cols], columns=categorical_cols)

# Standardizzazione delle variabili numeriche
scaler = StandardScaler()
X_numerical_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)

# Rimuovi le colonne numeriche originali da X_categorical_encoded
X_categorical_encoded = X_categorical_encoded.reset_index(drop=True)
X_numerical_scaled = X_numerical_scaled.reset_index(drop=True)

# Unisci le variabili numeriche scalate con quelle categoriche codificate
X_final = pd.concat([X_categorical_encoded, X_numerical_scaled], axis=1)

# All'attributo che abbiamo staccato, cioè quello che vogliamo prevedere, cioè il risultato,
# viene applicato il LabelEncoder che non è altro che l'assegnazione di un numero al risultato
# vittoria = 0, pareggio = 1, sconfitta = 2
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Suddividiamo il training set in training set e validation set, dove il validation set
# sarà il 20% di tutto il training set. Questo ci servirà per selezionare il modello migliore

test_x = X_final.tail(761)
test_y = y_encoded[-761:]
test_x['result'] = test_y
test_x.to_csv("test_set.csv", index=False)

# Si toglie l'ultima stagione
X_final = X_final.iloc[:-761]
y_encoded = y_encoded[:-761]

X_train, X_val, y_train, y_val = train_test_split(X_final, y_encoded, test_size=0.2, random_state=42)

best_model = create_network(
    X_train.shape[1],
    best_params["neurons_1layer"],
    best_params["neurons_2layer"],
    best_params["activation_function"]
)

# Compile the model with the best learning rate
best_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
    metrics=['accuracy']
)

# Train the model with the best hyperparameters
best_history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    verbose=1  # Show progress for the final training
)


# Evaluate the model on the validation data
final_val_loss = best_history.history['val_loss'][-1]
final_val_accuracy = best_history.history['val_accuracy'][-1]

print(f"Final validation loss with best hyperparameters: {final_val_loss}")
print(f"Final validation accuracy with best hyperparameters: {final_val_accuracy}")

print(best_history)
plt.figure(figsize=(8, 6))
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

best_model.save("best_model.h5")