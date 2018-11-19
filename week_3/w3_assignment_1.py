import pandas
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import data_management as dm
import matplotlib.pyplot as plt
import numpy as np


# #2
def polynomial_dataframe(feature, degree):  # feature is pandas.Series type
    poly_dataframe = pandas.DataFrame()
    poly_dataframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            poly_dataframe[name] = feature.apply(lambda x: x**power)
    return poly_dataframe


def get_estimated_model(data: list, input_models: list, y_param: str)-> RegressionResultsWrapper:
    y = data[y_param]
    x = sm.add_constant(data[input_models])
    model = sm.OLS(y, x)
    result = model.fit()
    return result


def print_coefficient(params: pandas.Series):

    for (key,value) in params.items():
        print('coefficient of {key} is: {coef}'.format(key=key, coef=value))


def create_polynomial(data: pandas.DataFrame, features: str, param_name: str, degree: int):
    poly_data: pandas.DataFrame = polynomial_dataframe(data[features], degree)
    poly_feature = poly_data.columns.values

    # #5
    poly_data[param_name] = data[param_name]

    return poly_data, poly_feature


def plot_polynomial(polynomial_data: pandas.DataFrame, predict_model, param_name: str):
    plot_feature_name = 'power_1'
    plt.plot(polynomial_data[plot_feature_name], polynomial_data[param_name], '.', polynomial_data[plot_feature_name],
             predict_model, '-')
    plt.show()


def show_result(poly_data: pandas.DataFrame, regression_model: RegressionResultsWrapper,
                predict_model: pandas.DataFrame, y_param: str, degree: int):

    poly_features = [i for i in poly_data.columns.values if i.startswith('power_')]

    # #7
    pred_model = regression_model.predict(sm.add_constant(predict_model[poly_features]))

    plot_polynomial(poly_data, pred_model, y_param)

    # #9
    print('-----coefficient of degree {deg}------'.format(deg=degree))

    print_coefficient(regression_model.params)


def get_polynomial_and_estimated_model(data: pandas.DataFrame, feature: str, y_param: str, degree: int):
    # #4
    poly_data, poly_features = create_polynomial(data, feature, y_param, degree)

    # #6
    model = get_estimated_model(poly_data, poly_features, y_param)

    show_result(poly_data, model, poly_data, y_param, degree)


def main():
    # #3
    sales = dm.get_w3_house_data()
    sales = sales.sort_values(['sqft_living', 'price'])

    features = 'sqft_living'
    y_param = 'price'

    # # #4
    get_polynomial_and_estimated_model(sales, features, y_param, 1)
    #
    # # #8
    get_polynomial_and_estimated_model(sales, features, y_param, 2)
    get_polynomial_and_estimated_model(sales, features, y_param, 3)

    # #9
    get_polynomial_and_estimated_model(sales, features, y_param, 15)

    # # #10
    # sub_model_1 = dm.get_w3_house_set_data(1)
    # sub_model_2 = dm.get_w3_house_set_data(2)
    # sub_model_3 = dm.get_w3_house_set_data(3)
    # sub_model_4 = dm.get_w3_house_set_data(4)
    #
    # # #11
    # get_polynomial_and_estimated_model(sub_model_1, features, y_param, 15)
    # get_polynomial_and_estimated_model(sub_model_2, features, y_param, 15)
    # get_polynomial_and_estimated_model(sub_model_3, features, y_param, 15)
    # get_polynomial_and_estimated_model(sub_model_4, features, y_param, 15)


if __name__ == "__main__":
    main()
