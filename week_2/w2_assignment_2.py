import numpy as np
from data_management import get_train_data, get_test_data
import helpers
import math


# #3
def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features

    features_frame = data_frame[features]
    features_matrix = np.array(features_frame)

    output_darray = data_frame[output]
    output_array = np.array(output_darray)

    return features_matrix, output_array


# #4
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


# #5
def feature_derivative(errors, feature):
    derivative = np.dot(errors, feature) * 2
    return derivative


# #6
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predictions = predict_outcome(feature_matrix, weights)

        errors = predictions - output

        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += (derivative ** 2)
            weights[i] -= (step_size * derivative)

        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return weights


def get_rss(predictions, output):
    residuals = output-predictions
    rss = (residuals**2).sum()
    return rss

def main():
    train_data = get_train_data()
    test_data = get_test_data()

    # #8
    simple_features = ['sqft_living']
    my_output = 'price'
    (simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
    initial_weights = np.array([-47000., 1.])
    step_size = 7e-12
    tolerance = 2.5e7

    simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

    # #9 Quiz
    print('sqft_living weight: {weight:10.2f}'.format(weight=simple_weights[1]))

    # #10
    (test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

    # #11
    test_predictions_model1 = predict_outcome(test_simple_feature_matrix, simple_weights)
    print('Pridicted price for the 1st house is: {price:10.0f}'.format(price=test_predictions_model1[0]))

    # #12
    test_rss = get_rss(test_predictions_model1, test_output)

    # #13
    model_features = ['sqft_living', 'sqft_living15']
    my_output = 'price'
    (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
    initial_weights = np.array([-100000., 1., 1.])
    step_size = 4e-12
    tolerance = 1e9

    weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

    # #14
    (test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
    test_predictions_model_2 = predict_outcome(test_feature_matrix, weights)

    # #15
    print('Pridicted price for the 1st house is: {price:10.0f}'.format(price=test_predictions_model_2[0]))

    # #16
    print('Actual price for the 1st house is: {price:10.0f}'.format(price=test_data['price'][0]))

    # #17
    test_rss_model_2  = get_rss(test_predictions_model_2,test_output)

    # #19
    lowest_index = helpers.get_lowest_index([test_rss,test_rss_model_2])
    print('The lowest rss is for Model-{index}'.format(index=lowest_index+1))


if __name__ == "__main__":
    main()
