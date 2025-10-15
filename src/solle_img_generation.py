import os
import mimetypes
import google.generativeai as genai
from .config import load_prompt_template


class imggenrator:
    def __init__(self):
        self.api_key = os.getenv("IMG_GEN_API_KEY")
        self.model_name = os.getenv("MODEL")
        if not self.api_key:
            raise ValueError("IMG_GEN_API_KEY not set in environment")
        if not self.model_name:
            raise ValueError("MODEL not set in environment")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, base_image_path: str, solana_logo_path: str = None, user_input_text: str = ""):
        # Charger l'image de base
        with open(base_image_path, "rb") as f:
            base_image = {
                "mime_type": mimetypes.guess_type(base_image_path)[0] or "application/octet-stream",
                "data": f.read(),
            }

        parts = [base_image]

        if solana_logo_path:
            with open(solana_logo_path, "rb") as f:
                logo_image = {
                    "mime_type": mimetypes.guess_type(solana_logo_path)[0] or "application/octet-stream",
                    "data": f.read(),
                }
            parts.append(logo_image)

        full_prompt = load_prompt_template(base_image_path, solana_logo_path, user_input_text)
        if not full_prompt:
            raise ValueError("load_prompt_template returned None or empty string")
        parts.append(full_prompt)

        result = self.model.generate_content(parts)
        if not result.candidates or not result.candidates[0].content.parts:
            raise RuntimeError("No content returned by the model")

        image_data = result.candidates[0].content.parts[0].data
        return [image_data]





