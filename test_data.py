from main import DataLoader, DataValidator


def test_data_validation():
    data = DataLoader.load_titanic_data()
    X, _ = DataLoader.preprocess_titanic_data(data)

    success, _ = DataValidator.validate_titanic_data(X)
    assert success
