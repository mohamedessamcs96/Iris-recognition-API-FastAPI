#bring in lightweight dependencies


from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd

app=FastAPI()


model=joblib.load('irismodel')


class ScoringItem(BaseModel):
    SepalLengthCm: float #1, 
    SepalWidthCm: float #0.01,
    PetalLengthCm: float #"Non-Manager",
    PetalWidthCm:float  #4.0



@app.post('/predict')
async def scoring_endpoint(item:ScoringItem):
    df=pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat=model.predict(df)
    return  {"Prediction":int(yhat)}


@app.get('/')
async def x():
    return  {"Hello","World!"}