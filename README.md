# Clasificador de Incidencias con SVM y Embeddings

## 1. Crear ambiente
conda env create -f environment.yml
conda activate ia_incidencias_svm

## 2. Colocar dataset
Copiar el CSV a:
data/tickets.csv

## 3. Entrenar
python -m app.train

## 4. Levantar API
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload

## 5. Probar
Abrir:
http://127.0.0.1:8000/docs