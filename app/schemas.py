from pydantic import BaseModel


class OCRResponse(BaseModel):
    id: str
    title: str
    description: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "n1",
                "title": "eSpill lluent",
                "description": "Presenta l'espill cara avall...",
            }
        }
