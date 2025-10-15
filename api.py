import os
import google.generativeai as genai
from .config import load_prompt_template

class ImgGenerator:
    def __init__(self):
        self.api_key = os.getenv("IMG_GEN_API_KEY")
        self.model_name = os.getenv("MODEL")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, base_image_path: str, solana_logo_path: str, user_prompt: str):
        full_prompt = load_prompt_template(base_image_path, solana_logo_path, user_prompt)
        if not full_prompt:
            raise ValueError("Le prompt généré est vide")
        result = self.model.generate_image(
            prompt=full_prompt,
            image_inputs=[base_image_path, solana_logo_path] if solana_logo_path else [base_image_path]
        )
        #adaptation de la generation d'image
        image_data = result[0].b64_image
        return [base64.b64decode(image_data)]

