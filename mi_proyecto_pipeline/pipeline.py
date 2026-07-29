from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def crear_preprocesador(caracteristicas_numericas, caracteristicas_categoricas):
    transformador_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformador_categorico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", transformador_numerico, caracteristicas_numericas),
            ("cat", transformador_categorico, caracteristicas_categoricas),
        ],
        remainder="drop",
    )


def crear_pipeline_completo(caracteristicas_numericas, caracteristicas_categoricas):
    preprocesador = crear_preprocesador(caracteristicas_numericas, caracteristicas_categoricas)
    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
    )


def entrenar_y_evaluar(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]

    caracteristicas_numericas = X.select_dtypes(include=["number"]).columns.tolist()
    caracteristicas_categoricas = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = crear_pipeline_completo(caracteristicas_numericas, caracteristicas_categoricas)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return pipeline, accuracy, report
