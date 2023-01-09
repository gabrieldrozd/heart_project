import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Wczytanie zbioru danych
data_frame = pd.read_csv("heart_failure_dataset.csv")
data_frame.columns
data_frame.shape
data_frame.head()

# Statystyki atrybutów z wczytanego zbioru danych
data_frame_attribute_stats = data_frame.describe().transpose()
np.savetxt("outcome_files/data_frame_attribute_stats.csv", data_frame_attribute_stats, delimiter=",")

# Normalizacja kolumn numerycznych
numeric_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']

data = data_frame[numeric_columns]

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Utworzenie nowego obiektu zwierającego znormalizowane dane oraz dane kategoryczne
normalized_data = np.concatenate(
    (normalized_data, data_frame[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']].values), axis=1)

np.savetxt("outcome_files/normalized_data.csv", normalized_data, delimiter=",")

# Podział zbioru danych na zbiór uczący i testowy
labels = data_frame['DEATH_EVENT'].values
X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2)

# Zapisanie zbiorów uczących i testowych do plików
np.savetxt("outcome_files/X_train.csv", X_train, delimiter=",")
np.savetxt("outcome_files/X_test.csv", X_test, delimiter=",")
np.savetxt("outcome_files/y_train.csv", y_train, delimiter=",")
np.savetxt("outcome_files/y_test.csv", y_test, delimiter=",")



# Define the model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# Utworzenie instancji klasyfikatora
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Use the classifier to predict labels for the test data
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=100, batch_size=32)
#
# # Evaluate the model on the test data
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
