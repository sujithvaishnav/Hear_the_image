from pydantic import BaseModel

class PredictResponse(BaseModel):
    caption: str
    text: str
