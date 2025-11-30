import pandas as pd
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import joblib
import mlflow


parampath = "params.yaml"
with open(parampath, "r") as f:
    params = yaml.safe_load(f)

processed_path = params["data"]["processed_path"]

train = pd.read_csv(processed_path + "train.csv")
test = pd.read_csv(processed_path + "test.csv")

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]

X_test = test.drop("Survived", axis=1)
y_test = test["Survived"]

dt_params = params["model_params"]
random_state = params["config"]["random_state"]

mlflow.sklearn.autolog()
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(params["mlflow"]["experiment_name"])

with mlflow.start_run() as run:
    model = DecisionTreeClassifier(random_state=random_state, **dt_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Создаём директорию и сохраняем модель
    model_path = params["model"]["path"]
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Модель сохранена в {model_path}")

    # Логирования MLflow
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_metric("accuracy", float(acc))
    mlflow.log_metric("precision", float(prec))
    mlflow.log_metric("recall", float(rec))
    mlflow.log_metric("f1", float(f1))

    mlflow.log_artifact(model_path)