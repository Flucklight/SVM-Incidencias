import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import SVM_MODEL_PATH, LOGREG_MODEL_PATH, LABEL_ENCODER_PATH, METADATA_PATH
from app.data_preprocess import limpiar_texto


class ClasificadorIncidencias:
    def __init__(self):
        self.clf = joblib.load(LOGREG_MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.metadata = joblib.load(METADATA_PATH)

        self.embedding_model = SentenceTransformer(
            self.metadata["embedding_model_name"]
        )

    @staticmethod
    def limpiar_texto(texto: str) -> str:
        if texto is None:
            return ""
        return limpiar_texto(texto)

    def predecir(self, texto: str) -> dict:
        texto_limpio = self.limpiar_texto(texto)

        if not texto_limpio:
            return {
                "texto_original": texto,
                "texto_procesado": "",
                "area_predicha": None,
                "confianza": 0.0,
                "top_3": []
            }

        vector = self.embedding_model.encode([texto_limpio], convert_to_numpy=True)

        pred = self.clf.predict(vector)[0]
        probs = self.clf.predict_proba(vector)[0]

        area = self.label_encoder.inverse_transform([pred])[0]

        indices = np.argsort(probs)[::-1]
        top_3 = [
            {
                "area": self.label_encoder.inverse_transform([idx])[0],
                "confianza": float(probs[idx]),
            }
            for idx in indices[:3]
        ]

        return {
            "texto_original": texto,
            "texto_procesado": texto_limpio,
            "area_predicha": area,
            "confianza": float(np.max(probs)),
            "top_3": top_3,
        }
