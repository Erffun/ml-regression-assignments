from statsmodels.regression.linear_model import RegressionResultsWrapper
import numpy
import statsmodels.api as sm
from data_management import get_train_data, get_test_data
import helpers


def add_new_values(data: list) -> list:
    data['bedrooms_squared'] = data['bedrooms'] ** 2
    data['bed_bath_rooms'] = data['bedrooms'] * data['bathrooms']
    data['log_sqft_living'] = numpy.log(data['sqft_living'])
    data['lat_plus_long'] = data['lat'] + data['long']
    return data


def print_new_vals_means(data: list):
    print('mean of bedrooms_squared is: {bs:10.2f}'.format(bs=data['bedrooms_squared'].mean()))
    print('mean of bed_bath_rooms is: {bedbath:10.2f}'.format(bedbath=data['bed_bath_rooms'].mean()))
    print('mean of log_sqft_living is: {logsq:10.2f}'.format(logsq=data['log_sqft_living'].mean()))
    print('mean of lat_plus_long is: {latlng:10.2f}'.format(latlng=data['lat_plus_long'].mean()))


def get_estimated_model(data: list, input_models: list, y_param: str)-> RegressionResultsWrapper:
    y = data[y_param]
    x = sm.add_constant(data[input_models])
    model = sm.OLS(y, x)
    result = model.fit()
    return result


def get_param_from_estimated_model(model: RegressionResultsWrapper, param_name: str)->float:
    params = model.params
    return params[param_name]


def get_rss(model: RegressionResultsWrapper, data: list, input_model: list, param_name: str)->float:
    prediction = model.predict(sm.add_constant(data[input_model]))
    residuals = data[param_name] - prediction
    rss = (residuals ** 2).sum()
    return rss


def main():
    train_data = get_train_data()
    test_data = get_test_data()

    # #3
    train_data = add_new_values(train_data)
    test_data = add_new_values(test_data)

    # #4 Quiz
    print_new_vals_means(test_data)

    model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
    model_2_features = model_1_features + ['bed_bath_rooms']
    model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

    model_1_estimated = get_estimated_model(train_data, model_1_features, 'price')
    model_2_estimated = get_estimated_model(train_data, model_2_features, 'price')
    model_3_estimated = get_estimated_model(train_data, model_3_features, 'price')

    # #6 Quiz
    model_1_coefficient = get_param_from_estimated_model(model_1_estimated, 'bathrooms')
    print('Model 1: coefficient of bedroom is: {coef:10.2f}'.format(coef=model_1_coefficient))

    # #7 Quiz
    model_2_coefficient = get_param_from_estimated_model(model_2_estimated, 'bathrooms')
    print('Model 2: coefficient of bedroom is: {coef:10.2f}'.format(coef=model_2_coefficient))

    # #9
    rss_list = dict()
    rss_list['Model 1'] = get_rss(model_1_estimated, train_data, model_1_features, 'price')
    rss_list['Model 2'] = get_rss(model_2_estimated, train_data, model_2_features, 'price')
    rss_list['Model 3'] = get_rss(model_3_estimated, train_data, model_3_features, 'price')

    min_rss = min(rss_list, key=rss_list.get)

    print('The lowest rss is for: {rss}'.format(rss=min_rss))

    # #10 Quiz
    model_1_rss = get_rss(model_1_estimated, test_data, model_1_features, 'price')
    model_2_rss = get_rss(model_2_estimated, test_data, model_2_features, 'price')
    model_3_rss = get_rss(model_3_estimated, test_data, model_3_features, 'price')
    lowest_index = helpers.get_lowest_index([model_1_rss, model_2_rss, model_3_rss])
    print('The lowest rss is for Test Model-{index}'.format(index=lowest_index + 1))


if __name__ == "__main__":
    main()
