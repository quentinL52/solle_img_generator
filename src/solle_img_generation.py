import os
import mimetypes
import google.generativeai as genai
from .config import load_prompt_template


class imggenrator:
    def __init__(self):
        self.api_key = os.getenv("IMG_GEN_API_KEY")
        self.model_name = os.getenv("MODEL")  # Default fallback
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, base_image_path: str, solana_logo_path: str, user_input_text: str):
        # Load the base image
        with open(base_image_path, "rb") as f:
            base_image = {
                "mime_type": mimetypes.guess_type(base_image_path)[0],
                "data": f.read(),
            }
        inputs = [base_image]
        if solana_logo_path:
            with open(solana_logo_path, "rb") as f:
                logo_image = {
                    "mime_type": mimetypes.guess_type(solana_logo_path)[0],
                    "data": f.read(),
                }
            inputs.append(logo_image)
        full_prompt = load_prompt_template(base_image_path, solana_logo_path, user_input_text)
        inputs.append(full_prompt)
        result = self.model.generate_content(
            inputs,
            generation_config={"response_mime_type": "image/png"},
        )
        image_data = result.candidates[0].content.parts[0].data
        return [image_data]


