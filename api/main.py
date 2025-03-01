from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# para correr: fastapi dev main.py
app = FastAPI()
model = joblib.load("assets/mejor_modelo.pkl")

results = {
    0:  "Draw",
    1: "Home Win",
    -1: "Away Win"
}


class Request(BaseModel):
    round: str
    homeClub: str
    awayClub: str


@app.post("/predict")
def read_root(request: Request):
    data = [
        {
            "Round": request.round,
            "Home Club": request.homeClub,
            "Away Club": request.awayClub
        }
    ]
    df = pd.DataFrame(data)
    predicciones = model.predict(df)
    return {"result": results[predicciones[0]]}
