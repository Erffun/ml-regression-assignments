import pandas

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}


def read_csv_file(filename: str)->list:
    data = pandas.read_csv('../csvFiles/{filename}.csv'.format(filename=filename), dtype=dtype_dict)
    return data


def get_train_data()->list:
    return read_csv_file('kc_house_train_data')


def get_test_data()->list:
    return read_csv_file('kc_house_test_data')


def get_w3_train_data()->list:
    return read_csv_file('wk3_kc_house_train_data')


def get_w3_test_data()->list:
    return read_csv_file('wk3_kc_house_test_data')


def get_w3_valid_data()->list:
    return read_csv_file('wk3_kc_house_valid_data')


def get_w3_house_set_data(set_number: int)->list:
    return read_csv_file('wk3_kc_house_set_{number}_data'.format(number=set_number))
