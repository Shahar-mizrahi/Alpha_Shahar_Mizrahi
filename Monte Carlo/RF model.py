import ee
from itertools import combinations
import csv
import numpy as np

# Initialize the Earth Engine library
ee.Authenticate()
ee.Initialize(project='alphaproject-442906')

NUM_OF_ITERATIONS = 10
LIMIT = None # testing small parts

dataset = ee.FeatureCollection("projects/ee-shahar/assets/data_1")

# List of all possible independent variables
all_independent_vars = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
dependent = 'GPP_DT_VUT_REF'  # Dependent variable
par = 'PAR' # additional parameter

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
csv_data = [["Bands",  "NRMSE%"]]

nrmse_list = list([])

# Iterate through combinations and calculate errors
for i, combination in enumerate(all_combinations):
    real_values = list([])
    predicted_values = list([])

    print(f"Processing combination {i + 1}/{len(all_combinations)}:",  ", ".join(combination))
    for j in range(NUM_OF_ITERATIONS):
        seed = j

        # Create a dataset with only the necessary features and add a random column for shuffling
        training_data = dataset.select(list(combination) + [dependent, par]).randomColumn('random_column', seed).sort('random_column')

        # Split the data into training and testing sets (80% training, 20% testing)
        split = 0.8
        total_size = training_data.size()
        training_size = ee.Number(total_size).multiply(split).floor()

        training_set = training_data.limit(training_size)
        test_set = training_data.limit(total_size.subtract(training_size), training_size)

        # Remove the random column from both training and testing sets
        training_set = training_set.select(list(combination) + [dependent, par])
        test_set = test_set.select(list(combination) + [dependent, par])

        # Define a Random Forest Regressor and train it on the training data
        variables_per_split = int(len(combination) ** 0.5)
        random_forest = ee.Classifier.smileRandomForest(numberOfTrees=100, variablesPerSplit=variables_per_split, minLeafPopulation = 2, maxNodes = 30, bagFraction = 0.9, seed=42).setOutputMode('REGRESSION')
        trained_model = random_forest.train(
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
csv_output_path = r"C:\Users\Cyber_User\Desktop\alpha_codes\code_results/RF_best_combination_results88882.csv"

# Add data to the CSV
for combination, nrmse_ in zip(all_combinations, nrmse_list):
    csv_data.append([", ".join(combination), str(round(nrmse_, 3)) + "%"])

# Write the data to the CSV file
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"CSV file has been saved to: {csv_output_path}")

