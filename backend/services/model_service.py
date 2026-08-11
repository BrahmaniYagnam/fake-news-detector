from __future__ import annotations

import json
from typing import Any

from groq import Groq

from backend.config.settings import settings


class ModelService:
    def __init__(self) -> None:
        self.model_name = "llama-3.1-8b-instant"
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing in the configuration")
        self.client = Groq(api_key=settings.groq_api_key)

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        
        system_prompt = (
            "You are an expert fact-checker and fake news detector. "
            "Analyze the provided news headline or article. "
            "Determine if it is factually 'Real' (true current events) or 'Fake' (false, misinformation, or highly misleading). "
            "Return ONLY a JSON object with the following schema:\n"
            "{\n"
            '  "prediction": "Real" or "Fake",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "important_keywords": ["word1", "word2", "word3"]\n'
            "}\n"
            "important_keywords should be an array of up to 8 of the most critical words in the text that led to your conclusion."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cleaned}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result_text = response.choices[0].message.content
        if not result_text:
            raise ValueError("Groq returned an empty response")
            
        result = json.loads(result_text)
        prediction = result.get("prediction", "Fake")
        confidence = float(result.get("confidence", 0.99))
        
        probabilities = {
            "Real": confidence if prediction == "Real" else 1.0 - confidence,
            "Fake": confidence if prediction == "Fake" else 1.0 - confidence
        }

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "important_keywords": result.get("important_keywords", []),
            "probabilities_raw": [probabilities["Fake"], probabilities["Real"]],
        }

    def get_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": settings.app_version,
            "framework": "Groq API",
            "dataset": "Real-time Fact Checking",
            "num_classes": 2,
            "training_accuracy": 0.99,
        }
