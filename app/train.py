import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

from app.config import (
    DATASET_PROCESSED_PATH,
    SVM_MODEL_PATH,
    LOGREG_MODEL_PATH,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)


if __name__ == "__main__":
    print("Cargando dataset...")
    df = pd.read_csv(DATASET_PROCESSED_PATH)

    print("\nCargando modelo de embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Generando embeddings...")
    X = embedding_model.encode(
        df["texto"].tolist(), show_progress_bar=True, convert_to_numpy=True
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["area"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nEntrenando SVM...")
    param_grid = {
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    clf = SVC(
        class_weight="balanced",
        probability=True,
    )
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\n====== Mejores parámetros ======")
    print(grid.best_params_)

    y_pred = best_model.predict(X_test)

    print("\n===== MÉTRICAS =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best_model, SVM_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(
        {
            "embedding_model_name": EMBEDDING_MODEL_NAME,
            "classes": list(label_encoder.classes_),
            "n_samples": len(df),
            "text_source": "subject + body",
            "label_source": "business_type",
            "language_filter": "es",
        },
        METADATA_PATH,
    )

    print("\nModelo SVM guardado correctamente.")
    print(f"Modelo: {SVM_MODEL_PATH}")
    print(f"Encoder: {LABEL_ENCODER_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    print("\nEntrenando Modelo de Regresión Logística...")
    param_grid = {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10]}
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=-1
    )
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\n====== Mejores parámetros ======")
    print(grid.best_params_)

    y_pred = best_model.predict(X_test)

    print("\n===== MÉTRICAS =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best_model, LOGREG_MODEL_PATH)

    print("\nModelo de Regresión Logística guardado correctamente.")
    print(f"Modelo: {LOGREG_MODEL_PATH}")

    joblib.dump(best_model, LOGREG_MODEL_PATH)