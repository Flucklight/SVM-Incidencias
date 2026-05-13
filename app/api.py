from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import ClasificadorIncidencias

app = FastAPI(title="Clasificador de Incidencias")

clasificador = ClasificadorIncidencias()


class TicketEntrada(BaseModel):
    comentario: str


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Servicio activo"}


@app.post("/clasificar")
def clasificar_ticket(data: TicketEntrada):
    return clasificador.predecir(data.comentario)