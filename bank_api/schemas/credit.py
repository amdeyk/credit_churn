from pydantic import BaseModel

class Credit(BaseModel):
    name: str
    api_version: str
    model_version: str
