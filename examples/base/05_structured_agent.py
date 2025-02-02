'''Example of an agent with structured outputs'''
import json
import asyncio
from babelgraph.core.agent import BaseAgent
from pydantic import BaseModel, Field


class Car(BaseModel):
    """Example model for structured car information"""
    make: str = Field(..., description="The manufacturer of the car")
    model: str = Field(..., description="The model name of the car")
    year: int = Field(..., description="The year the car was manufactured")

json_agent = BaseAgent(json_mode=True)

structured_agent = BaseAgent(response_model=Car)


async def main():
    print("Basic JSON Response Agent")
    response = await json_agent._step("Tell me about machine learning.")
    print(f"JSON Mode Only:\n{json.dumps(response, indent=2)}")
    print("-"*100)
    print("-"*100)
    print("Structured Response using Response Model")
    response = await structured_agent._step("Tell me about the Toyota Camry.")
    # Convert Pydantic model to dict before JSON serialization
    response_dict = response.model_dump()
    print(f"Structured Output:\n{json.dumps(response_dict, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
