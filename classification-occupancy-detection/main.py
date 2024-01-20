import os
import warnings
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 200)


def get_filepaths() -> tuple:
    current_dir = os.getcwd()
    datasets_dir = current_dir + '/dataset'

    fp_train = datasets_dir + '/train.csv'
    fp_dev = datasets_dir + '/dev.csv'
    fp_test = datasets_dir + '/test.csv'
    return fp_train, fp_dev, fp_test


def load_dataset(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"{filepath.split(sep='/')[-1]} is loaded. Shape: {df.shape}")
    return df


def grab_columns(df: pd.DataFrame, car_th=20, num_cat_th=10, verbose=True) -> dict:
    cat_cols = [c for c in df.columns if df[c].dtype == 'O']
    num_cols = [c for c in df.columns if df[c].dtype != 'O']

    num_but_cat_cols = [c for c in num_cols if df[c].nunique() < num_cat_th]
    num_cols = [c for c in num_cols if c not in num_but_cat_cols]

    cat_but_car = [c for c in cat_cols if df[c].nunique() > car_th]
    cat_cols = [c for c in cat_cols + num_but_cat_cols if c not in cat_but_car]
    dct = {'categorical': cat_cols, 'numerical': num_cols,
           'cardinality': cat_but_car, 'categorical_as_numeric': num_but_cat_cols}

    if verbose:
        print("----------------------- First Sight ------------------------")
        print(f"Observation count: {df.shape[0]} \t | \t Variables : {df.shape[1]}")
        print("----------------------- Summary of Column Count ------------------------")
        print(f"Categorical cols: {len(cat_cols)}")
        print(f"Numerical Cols: {len(num_cols)}")
        print(f"High Cardinality Cols. {len(cat_but_car)}")
        print(f"Categorical as Numeric: {len(num_but_cat_cols)}")
        print("----------------------- Summary of Column Name ------------------------")
        print(f"Categorical cols: {cat_cols}")
        print(f"Numerical Cols: {num_cols}")
        print(f"High Cardinality Cols: {cat_but_car}")
        print(f"Categorical as Numeric: {num_but_cat_cols}")
        print("\n[info] * Dictionary Keys:", list(dct.keys()))

    return dct


def show_summary_table(df: pd.DataFrame, columns: List[str] = None) -> None:
    if columns is None:
        columns = df.columns.tolist()

    print("Observation : ", df.shape[0], " | Variables:", df.shape[1])
    df = pd.DataFrame([{'col': col,
                        'col_type': df[col].dtype,
                        'n_unique': df[col].nunique(),
                        'unique_values': df[col].unique(),
                        'n_missing': df[col].isnull().sum()}
                       for col in columns]) \
        .sort_values('n_unique', ascending=False) \
        .reset_index(drop=True)
    print(df, end="\n\n")


def show_feature_summary(df: pd.DataFrame, col: str, target: str) -> None:
    print(f"------------ Summary :: {col} -------------\n")
    df = pd.concat([df[col].value_counts(ascending=True),
                    df[col].value_counts(normalize=True, ascending=True),
                    df.groupby(col)[target].mean()],
                   keys=['n_observation', 'observation_ratio', 'target_mean'],
                   names=['Columns'],
                   axis=1)
    print(df, end="\n\n")


def get_outliers_columns(df: pd.DataFrame, cols: List[str] = None) -> List[str]:
    if cols is None:
        cols = df.columns.tolist()

    outlier_cols = [col for col in cols if df[col].dtype in ['int', 'float'] and df[col].nunique() > 10]
    return outlier_cols


def calculate_iqr_boundaries(df: pd.DataFrame, col: str, q1_ratio: float, q3_ratio: float) -> tuple:
    q1 = df[col].quantile(q1_ratio)
    q3 = df[col].quantile(q3_ratio)
    iqr = q3 - q1
    low_limit = q1 - (1.5 * iqr)
    up_limit = q3 + (1.5 * iqr)
    return low_limit, up_limit


def check_outliers(df: pd.DataFrame, q1_ratio: float, q3_ratio: float,
                   cols: List[str] = None) -> None:
    if cols is None:
        cols = get_outliers_columns(df)

    outliers = []
    for col in cols:
        low_limit, up_limit = calculate_iqr_boundaries(df, col, q1_ratio, q3_ratio)
        df_outlier = df[(df[col] < low_limit) | (df[col] > up_limit)]
        outliers.append([col, df_outlier.shape[0], df_outlier.shape[0] / df.shape[0]])

    outlier_summary = pd.DataFrame(data=outliers, columns=['column', 'n_outlier', 'n_ratio'])
    print(outlier_summary)


def fill_outliers_with_boundaries(df: pd.DataFrame, q1_ratio: float, q3_ratio: float, cols: List[str] = None):
    df_temp = df.copy()
    if cols is None:
        cols = get_outliers_columns(df)

    for col in cols:
        low_limit, up_limit = calculate_iqr_boundaries(df, col, q1_ratio, q3_ratio)
        is_outlier = df[(df[col] < low_limit) | (df[col] > up_limit)].any(axis=None)
        if is_outlier:
            df_temp.loc[df_temp[col] > up_limit, col] = up_limit
            df_temp.loc[df_temp[col] < low_limit, col] = low_limit
    return df_temp


def fill_missing_with_mean(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    df_temp = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].isna()]

    for col in cols:
        idx = df_temp[df_temp[col].isna()].index
        df_temp.loc[idx, col] = df_temp[col].mean()

    return df_temp


def set_season(month: int) -> int:
    if month <= 3:
        return -1  # "Winter"
    elif month <= 6:
        return 1  # "Spring"
    elif month <= 9:
        return 2  # "Summer"
    else:
        return 0  # "Autumn"


def set_day_status(hour: int) -> str:
    # :{minute}
    if hour <= 3:
        return "early-morning"
    if hour <= 9:
        return "morning"
    elif hour <= 13:
        return "before-noon"
    elif hour <= 17:
        return "afternoon"
    elif hour <= 21:
        return "evening"
    else:
        return "night"


def extract_info_from_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d %H:%M:%S')
    df["season"] = df["date"].apply(lambda row: set_season(row.month))
    df["month"] = df["date"].apply(lambda row: row.month)
    df["day_status"] = df["date"].apply(lambda row: set_day_status(hour=row.hour))
    df.drop(columns=["date"], inplace=True)
    return df


def generate_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp/co2"] = df["Temperature"] / df["CO2"]
    df["humidity*temp"] = df["Temperature"] * df["Humidity"]
    df["light/temp"] = df["Light"] * df["Temperature"]
    df["temp*season"] = df["Temperature"] * df["season"]
    df["humidity*season"] = df["Humidity"] * df["season"]
    return df


def get_ohe_cols(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if df[col].dtype == 'O' and 2 < df[col].nunique() < 30]
    return cols


def one_hot_encoder(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    if cols is None:
        cols = get_ohe_cols(df)

    X = df[cols]
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    ohe.fit(X)

    df_ohe = pd.DataFrame(ohe.transform(X).astype(int), columns=ohe.get_feature_names_out())
    df_final = pd.concat([df, df_ohe], axis=1).drop(columns=ohe.feature_names_in_, axis=1)

    return df_final


def label_encoder(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == 'O' and df[c].nunique() < 30]

    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def split_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    X = df.drop(columns=['Occupancy'])
    y = df[['Occupancy']]
    return X, y


def apply_scaling(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    cols_dct = grab_columns(df=df, car_th=20, num_cat_th=30, verbose=False)
    num_cols = cols_dct.get('numerical')
    data_scaled = scaler.transform(df[num_cols])
    df_scaled = pd.DataFrame(data=data_scaled, columns=num_cols)

    for col in df_scaled.columns:
        df[col] = df_scaled[col]

    return df


# Model & Training & Evaluation
class MyNetModel(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_features, out_features=12)
        self.hidden_layer1 = nn.Linear(in_features=12, out_features=4)
        self.output_layer = nn.Linear(in_features=4, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.hidden_layer1(out)
        out = self.relu(out)
        out = self.output_layer(out)
        out = self.sigmoid(out)
        return out


def convert_to_torch_tensor(X, y) -> tuple:
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
    print("X.shape:", X.shape, "\ty.shape:", y.shape)
    return X, y


def train(X, y, n_epochs=20, lr=1e-3, batch_size=10):
    model = MyNetModel(input_features=X.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_pred = model(X_batch)
            y_batch = y[i:i + batch_size]
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss:.3f}')

    return model


def predict_and_evaluate(model: MyNetModel, X, y, label: str):
    with torch.no_grad():
        y_pred = model(X)

    accuracy = (y_pred.round() == y).float().mean()
    print(f"{label} Accuracy {accuracy:.3f}")
    return accuracy

def main():
    # Load Data
    fp_train, fp_dev, fp_test = get_filepaths()
    df_train = load_dataset(fp_train)
    df_dev = load_dataset(fp_dev)
    df_test = load_dataset(fp_test)
    
    # First Insight about Data
    show_summary_table(df_train)
    show_summary_table(df_dev)
    show_summary_table(df_test)
    
    # Check outliers
    check_outliers(df=df_train, q1_ratio=.05, q3_ratio=.95)
    check_outliers(df=df_dev, q1_ratio=.05, q3_ratio=.95)
    check_outliers(df=df_test, q1_ratio=.05, q3_ratio=.95)
    
    # Fill Outliers
    df_train_filled = fill_outliers_with_boundaries(df=df_train, q1_ratio=.05, q3_ratio=.95)
    df_dev_filled = fill_outliers_with_boundaries(df=df_dev, q1_ratio=.05, q3_ratio=.95)
    df_test_filled = fill_outliers_with_boundaries(df=df_test, q1_ratio=.05, q3_ratio=.95)
    
    # Look data summary again after fill outliers
    show_summary_table(df_train_filled)
    show_summary_table(df_dev_filled)
    show_summary_table(df_test_filled)
    
    # Fill missing values
    df_train_filled = fill_missing_with_mean(df_train_filled, cols=["CO2", "Temperature"])
    show_summary_table(df_train_filled)
    
    # Convert date column & generate new features
    df_train_final = extract_info_from_date(df=df_train_filled)
    df_train_final = generate_new_features(df=df_train_final)
    df_dev_final = extract_info_from_date(df=df_dev_filled)
    df_dev_final = generate_new_features(df=df_dev_final)
    df_test_final = extract_info_from_date(df=df_test_filled)
    df_test_final = generate_new_features(df=df_test_final)
    
    #Final insights
    show_summary_table(df_train_final)
    show_summary_table(df_dev_final)
    show_summary_table(df_test_final)
    
    # I use LabelEncoder Instead of OneHotEncoder. Because in training steps there are  dimensonality
    # problem for feature space between datasets
    df_train_encoded = label_encoder(df=df_train_final)  # one_hot_encoder(df=df_train_final)
    df_dev_encoded = label_encoder(df=df_dev_final)  # one_hot_encoder(df=df_dev_final)
    df_test_encoded = label_encoder(df=df_test_final)  # one_hot_encoder(df=df_test_final)
    
    # Scale datasets
    df_train_X, df_train_y = split_dataset(df_train_encoded)
    num_train_cols = grab_columns(df=df_train_X, num_cat_th=30, car_th=20).get('numerical')
    scaler_init = StandardScaler()
    scaler_init.fit(df_train_X[num_train_cols])
    df_train_X_scaled = apply_scaling(df=df_train_X, scaler=scaler_init)
    df_dev_X, df_dev_y = split_dataset(df_dev_encoded)
    df_dev_X_scaled = apply_scaling(df=df_dev_X, scaler=scaler_init)
    df_test_X, df_test_y = split_dataset(df_test_encoded)
    df_test_X_scaled = apply_scaling(df=df_test_X, scaler=scaler_init)
    
    # Train & Evaluate
    X_train_torch, y_train_torch = convert_to_torch_tensor(X=df_train_X_scaled, y=df_train_y)
    X_dev_torch, y_dev_torch = convert_to_torch_tensor(X=df_dev_X_scaled, y=df_dev_y)
    X_test_torch, y_test_torch = convert_to_torch_tensor(X=df_test_X_scaled, y=df_test_y)
    
    model = train(X=X_train_torch, y=y_train_torch, n_epochs=20, lr=1e-3, batch_size=10)
    acc_train = predict_and_evaluate(model, X=X_train_torch, y=y_train_torch,label='Train')
    acc_dev = predict_and_evaluate(model, X=X_dev_torch, y=y_dev_torch,label='Dev')
    acc_test = predict_and_evaluate(model, X=X_test_torch, y=y_test_torch,label='Test')
    
    """
    Train Accuracy 0.989
    Dev Accuracy 0.978
    Test Accuracy 0.981
    """
    
if __name__ == '__main__':
    main()
