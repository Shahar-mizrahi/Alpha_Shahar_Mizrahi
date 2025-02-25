import ee
from itertools import combinations
import csv
import numpy as np

# Initialize the Earth Engine library
ee.Authenticate()
ee.Initialize(project='alphaproject-442906')

LIMIT = None # testing small parts
par = 'PAR' # additional parameter


# set up all of the data
data_2019 = ee.FeatureCollection("projects/ee-shahar/assets/data_1_2019")
data_2020 = ee.FeatureCollection("projects/ee-shahar/assets/data_1_2020")
data_2021 = ee.FeatureCollection("projects/ee-shahar/assets/data_1_2021")

data_2019_2020 = ee.FeatureCollection("projects/ee-shahar/assets/data_1_2019_2020")
data_2019_2021 = ee.FeatureCollection("projects/ee-shahar/assets/data_1_2019_2021")
data_2020_2021 = ee.FeatureCollection("projects/ee-shahar/assets/data_1_2020_2021")

training_datasets = [data_2020_2021, data_2019_2021, data_2019_2020]
testing_datasets = [data_2019, data_2020, data_2021]

# List of all possible independent variables and the dependent variable
all_independent_vars = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
dependent = 'GPP_DT_VUT_REF'


def get_combinations(array):
    """
    Generate all possible combinations of the independent variables.
    """
    result = []
    for r in range(1, len(array) + 1):
        result.extend(combinations(array, r))
    return result


def get_predictions(feature):
    """
    Calculate the error between predicted and actual dependent variable values.
    """
    prediction = feature.getNumber('classification')
    actual = feature.getNumber(dependent)
    error = actual.subtract(prediction).abs()
    return feature.set({'error': error, 'predicted': prediction, 'real': actual})


# Generate all possible combinations of the independent variables
all_combinations = get_combinations(all_independent_vars)
csv_data = [["Bands", "NRMSE%"]]

# csv_data2 = [["real", "predicted"]]
nrmse_list = list([])

# Iterate through combinations and calculate errors
for i, combination in enumerate(all_combinations):
    real_values = list([])
    predicted_values = list([])

    print(f"Processing combination {i + 1}/{len(all_combinations)}: {', '.join(combination)}")
    for training, testing in zip(training_datasets, testing_datasets):

        # set up the traning and testing data
        training_set = training.select(list(combination) + [dependent, par])
        test_set = testing.select(list(combination) + [dependent, par])

        # Define an SVM Regressor and train it on the training data
        svm = ee.Classifier.libsvm(svmType= 'EPSILON_SVR', kernelType= 'RBF', shrinking= True, cost = 1, lossEpsilon = 5 ,terminationEpsilon = 0.01, gamma= 0.9).setOutputMode('REGRESSION')
        trained_model = svm.train(
            features=training_set,
            classProperty=dependent,
            inputProperties=list(combination) + [par]
        )

        # Make predictions on the test data
        predictions = test_set.classify(trained_model)

        # Compare predictions to actual values and add an error column
        predictions_with_error = predictions.map(get_predictions)

        # Extract real and predicted GPP values and the errors for this iteration
        real_values_iter = predictions_with_error.aggregate_array('real').getInfo()
        predicted_values_iter = predictions_with_error.aggregate_array('predicted').getInfo()

        # Append the values to the respective lists
        real_values.extend(real_values_iter)
        predicted_values.extend(predicted_values_iter)

    # numpy moudle
    real_values = np.array(real_values)
    predicted_values = np.array(predicted_values)
    
    # calc RMSE
    rmse = np.sqrt(np.mean((real_values - predicted_values) ** 2))
    # calc NRMSE%
    mean_true_values = np.mean(real_values)
    nrmse = (rmse / mean_true_values) * 100
    print(str(round(nrmse, 3)) + "%")
    nrmse_list.append(nrmse)
    
    # Break early for testing purposes
    if i == LIMIT:
        break

lowest_index = nrmse_list.index(min(nrmse_list))
print("lowest nrmse:", str(round(nrmse_list[lowest_index], 3)) + "%")
print("the bands:", ", ".join(all_combinations[lowest_index]))

# Define output CSV file path
csv_output_path = r"C:\Users\Cyber_User\Desktop\alpha_codes\code_results/SVM_best_combination_results.csv"

# Add data to the CSV
for combination, nrmse_ in zip(all_combinations, nrmse_list):
    csv_data.append([", ".join(combination), str(round(nrmse_, 3)) + "%"])

# Write the data to the CSV file
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"CSV file has been saved to: {csv_output_path}")
