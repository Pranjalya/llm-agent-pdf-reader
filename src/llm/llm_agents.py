from abc import ABC, abstractmethod
from typing import Any
import logging

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMAgentInterface(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

class GeminiLLMAgent(LLMAgentInterface):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def query(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error querying Gemini LLM: {e}")
            return "Error processing the question."

class OpenAILLMAgent(LLMAgentInterface):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def query(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error querying OpenAI LLM: {e}")
            return "Error processing the question."
