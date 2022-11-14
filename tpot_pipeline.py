import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import coremltools

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('dataset/dataframe_hrv_test.csv', delimiter=',', dtype=np.float64)
# print(tpot_data['stress'].size())

for i in range(len(tpot_data['stress'])):
    if(tpot_data['stress'][i] < 0.4):
        tpot_data['stress'][i] = 0
    elif(tpot_data['stress'][i] < 0.6):
        tpot_data['stress'][i] = 1
    elif(tpot_data['stress'][i] < 0.8):
        tpot_data['stress'][i] = 2
    elif(tpot_data['stress'][i] <= 1):
        tpot_data['stress'][i] = 3

print(tpot_data['stress'][44])

features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('stress'), axis=1)

print(features)

training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['stress'], random_state=42)

# print(training_target.)

exported_pipeline = KNeighborsClassifier(n_neighbors=4, weights="distance")

# training the model
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# input_features just shows as sample
input_features = ["heartRate", "SDNN"]
output_features = "stress"

model = coremltools.converters.sklearn.convert(exported_pipeline, input_features, output_features)
model.save("HrvStressClassifier.mlmodel")
