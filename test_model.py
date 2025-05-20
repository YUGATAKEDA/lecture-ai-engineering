from main import DataLoader, ModelTester
from sklearn.model_selection import train_test_split


def test_model_performance():
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = ModelTester.train_model(X_train, y_train)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    assert metrics["accuracy"] >= 0.75
    assert metrics["inference_time"] < 1.0
