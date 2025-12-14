from ml_model import load_data
from ml_model import load_data, preprocess_data
from ml_model import load_data, preprocess_data, train_model
from ml_model import load_data, preprocess_data, train_model, predict



def test_load_data():
    df = load_data()
    assert not df.empty
    assert 'Outcome' in df.columns

def test_preprocess_data_shape():
    df = load_data()
    X, y, imputer, scaler = preprocess_data(df)

    assert X.shape[0] == len(df)
    assert X.shape[1] == len(df.columns) - 1


def test_model_training():
    df = load_data()
    X, y, _, _ = preprocess_data(df)
    model, X_test, y_test = train_model(X, y)

    assert model is not None
    assert len(X_test) == len(y_test)


def test_prediction_output():
    df = load_data()
    X, y, imputer, scaler = preprocess_data(df)
    model, _, _ = train_model(X, y)

    sample_input = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    pred = predict(model, scaler, imputer, sample_input)

    assert pred[0] in [0, 1]