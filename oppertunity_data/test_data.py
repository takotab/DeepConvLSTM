from . import get_data


def test_get_data():
    import config
    features_train, y_train, features_test, y_test = get_data(config)

    assert type(features_train) is dict
    assert type(features_test) is dict
    shapes = {}
    for features, y in [(features_train, y_train), (features_test, y_test)]:
        print(y.shape)
        for key in features:
            print(key, features[key].shape)
            if key in shapes:
                assert features[key].shape[1:] == shapes[key][1:], shapes
            else:
                shapes[key] = features[key].shape

            assert features[key].shape[0] == y.shape[0]
