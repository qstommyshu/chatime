from pydantic import BaseModel, Field
from typing import List

class StructuredAnswer(BaseModel):
    """Response model for the chat API"""
    answer: str = Field(description="The answer to the user's question.")
    sources: List[str] = Field(description="A list of source URLs used to answer the question.") 