import os
import mimetypes
import google.generativeai as genai
from google.generativeai import types
from .config import load_prompt_template


class imggenrator:
    def __init__(self):
        self.api_key = os.getenv("IMG_GEN_API_KEY")
        self.model_name = os.getenv("MODEL", "gemini-1.5-flash")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, base_image_path: str, solana_logo_path: str, user_input_text: str):
        """Generate image(s) based on a base image, optional logo, and user text prompt."""
        contents_parts = []
        with open(base_image_path, "rb") as f:
            image_data = f.read()
        image_mime_type = mimetypes.guess_type(base_image_path)[0]
        contents_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))
        if solana_logo_path:
            with open(solana_logo_path, "rb") as f:
                image_data = f.read()
            image_mime_type = mimetypes.guess_type(solana_logo_path)[0]
            contents_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))
        full_prompt = load_prompt_template(base_image_path, solana_logo_path, user_input_text)
        contents_parts.append(types.Part.from_text(text=full_prompt))
        contents = [types.Content(role="user", parts=contents_parts)]
        generated_images_data = []
        for chunk in self.model.generate_content_stream(
            contents=contents,
            generation_config={"response_mime_type": "image/png"},
        ):
            if chunk.candidates:
                for candidate in chunk.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            generated_images_data.append(part.inline_data.data)
                        elif hasattr(part, "text") and part.text:
                            print(part.text)

        return generated_images_data

