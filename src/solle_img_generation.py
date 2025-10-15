import os
import google.generativeai as genai
import mimetypes 
from .config import load_prompt_template
from PIL import Image # Utilisé pour charger les images

class imggenrator:
    def __init__(self):
        self.api_key = os.getenv("IMG_GEN_API_KEY")
        self.model_name = os.getenv("MODEL")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, base_image_path: str, solana_logo_path: str = None, user_input_text: str = ""):
        try:
            base_image_pil = Image.open(base_image_path)
            parts = [base_image_pil]
        except FileNotFoundError:
            raise RuntimeError(f"Fichier de base non trouvé : {base_image_path}")
        if solana_logo_path:
            try:
                logo_image_pil = Image.open(solana_logo_path)
                parts.append(logo_image_pil)
            except FileNotFoundError:
                print(f"Attention : Fichier logo non trouvé, ignoré : {solana_logo_path}")
        full_prompt = load_prompt_template(base_image_path, solana_logo_path, user_input_text)
        parts.append(full_prompt)
        result = self.model.generate_content(parts)
        
        if not result.candidates or not result.candidates[0].content.parts:
            raise RuntimeError("No content returned by the model")
        image_data = result.candidates[0].content.parts[0].data
        return [image_data]
