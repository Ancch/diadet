from pydantic import BaseModel

class DiabetesInput(BaseModel):
    Age: float
    Gender: int   # 1 = Male, 0 = Female
    BMI: float
    BloodPressure: float
    Insulin: float
    Glucose: float
    DiabetesPedigreeFunction: float
