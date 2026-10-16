from pydantic import BaseModel, Field


class RentRequest(BaseModel):
    locality: str
    city: str
    area: float = Field(gt=0, le=10000)
    beds: int = Field(ge=1, le=10)
    bathrooms: int = Field(ge=0, le=10)
    furnishing: str


class RentResponse(BaseModel):
    rent: float
    low: float
    high: float
