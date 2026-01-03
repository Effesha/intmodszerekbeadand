from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# csv struktura atalakitasa
# sajnos a kaggle-en talalhato "california-housing-prices" strukturaja 
# nem teljesen egyezik meg a sklearn fetch_california_housing() strukturajaval,
# igy kisebb modositasok szuksegesek:
#   1) konverzio kezeles
#   2) NaN kezeles
def prepare_data_and_target(df):
    if 'ocean_proximity' in df.columns:
        df = df.drop(columns=['ocean_proximity']) # 1) string->float konverzio sikertelen lesz, el kell tavolitani

    df = df.fillna(df.median(numeric_only=True)) # 2) NaN ertekek talalhatok a csv-ben, melyek miatt szinten elszallhat a program
    x = df.drop(columns=['median_house_value']).values
    y = df['median_house_value'].values
    return x, y, list(df.drop(columns=['median_house_value']).columns)

def main():
    # a csv fajlt helyezzuk el a rootban
    df = pd.read_csv("housing.csv") # source: https://www.kaggle.com/datasets/camnugent/california-housing-prices

    # adatok elokeszitese
    x, y, feature_names = prepare_data_and_target(df)

    # Kért információk kiírása
    print("Dataset: housing.csv (https://www.kaggle.com/datasets/camnugent/california-housing-prices)")
    print(f" - mintak szama: {x.shape[0]}")
    print(f" - jellemzok szama: {x.shape[1]}")
    print(f" - jellemzok neve: {feature_names}")
    print()

    # adatok felosztasa tanito es teszt reszre a feladatban megadott ertekek szerint
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print("Train/Test felosztás:")
    print(f" - x_train shape: {x_train.shape}")
    print(f" - x_test shape:  {x_test.shape}")
    print()

    # baseline model: mindig a tanito halmazon mert atlagos celerteket josolja minden mintára
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    y_pred_base = baseline.predict(x_test)

    # mae: mean absolute error, atlagolja az elorejelzes es a valos ertek abszolut kulonbseget
    mae_base = mean_absolute_error(y_test, y_pred_base)

    # mse: mean squared error, atlagolja az elorejelzes es a valos ertek kulonbsegenek negyzetet
    mse_base = mean_squared_error(y_test, y_pred_base)

    # rmse: root mean squared error, atlagolja az elorejelzes es a valos ertek kozotti negyzetes hibat
    rmse_base = sqrt(mse_base)

    # r2: magyarazott variancia aranya, megmutatja, mennyivel pontosabb a modell, mintha mindig csak az atlagot hasznalnank elorejelzeskent
    r2_base = r2_score(y_test, y_pred_base)

    print("Baseline:")
    print(f" - MAE = {mae_base:.4f}")
    print(f" - MSE = {mse_base:.4f}")
    print(f" - RMSE = {rmse_base:.4f}")
    print(f" - R2 = {r2_base:.4f}")
    print()

    # skalazas: mivel a KNN tavolsag-alapu modszer, ezert a feature-ok skalazasa kotelezo, 
    # kulonben a nagy skalaju jellemzok elnyomjak a tobbit, ami felrevezeto eredmenyhez vezethet
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # KNN regresszio (k = 5)
    knn = KNeighborsRegressor(n_neighbors=5) # minden elorejelzeshez az 5 legkozelebbi szomszed hasznaalta
    knn.fit(x_train_scaled, y_train)
    y_pred_knn = knn.predict(x_test_scaled)

    # KNN kiertekelese ugyanugy, mint baseline eseten, csak most a knn elorejelzeseit hasznalva
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    rmse_knn = sqrt(mse_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    print("KNN:")
    print(f" - MAE = {mae_knn:.4f}")
    print(f" - MSE = {mse_knn:.4f}")
    print(f" - RMSE = {rmse_knn:.4f}")
    print(f" - R2 = {r2_knn:.4f}")
    print()

    # osszehasonlitas
    print("Összehasonlítás: baseline vs. knn")
    print(f" - MAE csökkenés (baseline -> knn): {mae_base - mae_knn:.4f}")
    print(f" - MSE csökkenés (baseline -> knn): {mse_base - mse_knn:.4f}")
    print(f" - RMSE csökkenés (baseline -> knn): {rmse_base - rmse_knn:.4f}")
    print(f" - R2 változás (baseline -> knn): {r2_knn - r2_base:.4f}")

main()

# kerdesek:
# javult-e a KNN a baseline-hoz kepest: a fenti szamok alapjan eldontheto,
# hogy javult a KNN (MAE, MSE és RMSE csökkenése jelzi a javulást).

# mit mutat az R2: a baseline R2-je azt mutatja, hogy semmit nem tanul az adatokbol (R2 = -0.0002).
# A KNN R2-je azt mutatja, hogy a KNN kepes volt mintazatokat megtanulni az adatokbol (R2 = 0.7046),
# es a baseline-hoz kepest ertelmes elorejelzest ad.

# miert fontos a skalazas: a KNN tavolsagokat szamol a mintak kozott. 
# Ha a jellemzok kulonbozo nagysagrenduek, akkor a nagyobb skalaju valtozok aranytalanul nagy hatast
# gyakorolnak a tavolsagra, mellyel eltorzithatjak az eredmenyt.
