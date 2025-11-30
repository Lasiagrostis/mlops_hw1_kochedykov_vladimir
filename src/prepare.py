import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

parampath = "params.yaml"
with open(parampath, "r") as f:
    params = yaml.safe_load(f)

data = pd.read_csv(params["data"]["raw_path"])

# Обработка пропущенных значений и кодирование категориальных признаков
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0]) \
    .map({"S": 0, "C": 1, "Q": 2}).astype(int)
data["Sex"] = data["Sex"].map({"male": 0, "female": 1}).astype(int)

# Оставляем только нужные столбцы
data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# Конфиг для разделения данных
test_size = params["config"]["split_ratio"]
random_state = params["config"]["random_state"]

train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["Survived"])


output_dir = params["data"]["processed_path"]
os.makedirs(output_dir, exist_ok=True)
train_data.to_csv(output_dir + "train.csv", index=False)
test_data.to_csv(output_dir + "test.csv", index=False)