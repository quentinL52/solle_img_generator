from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import io
from PIL import Image
import base64 # Importation conservée bien qu'inutilisée dans le code fourni
from src.solle_img_generation import imggenrator
from prompt_from_tweet.main import generate_image_prompt_from_tweet

APP_TITLE = "Solle Image Generation API"
INPUT_IMAGE_PATH = "solle_base.jpg"
SOLANA_LOGO_PATH = "solana_logo.png"

app = FastAPI(title=APP_TITLE)

class TweetRequest(BaseModel):
    tweet: str

@app.post("/generate-image", 
          summary="solle img generator from a tweet",
          response_description="Image PNG generated")
def generate_image(request: TweetRequest) -> Response:
    """
    take a tweet in entry turn it into a prompt then generate an image from it !
    """
    tweet_text = request.tweet.strip()
    
    # Validation du texte du tweet
    if not tweet_text:
        # Code d'erreur 400 pour mauvaise requête (Bad Request)
        raise HTTPException(status_code=400, detail="Le tweet ne peut pas être vide.")

    try:
        prompt_img = generate_image_prompt_from_tweet(tweet_text)
        model = imggenrator()
        generated_images = model.generate(
            base_image_path=INPUT_IMAGE_PATH, 
            solana_logo_path=SOLANA_LOGO_PATH, 
            user_input_text=prompt_img
        )
        if not generated_images or not generated_images[0]:
            # Code d'erreur 500 pour un problème côté serveur (Internal Server Error)
            raise HTTPException(status_code=500, detail="Aucune image générée ou données d'image invalides.")
        
        img_data = generated_images[0]
        image = Image.open(io.BytesIO(img_data))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur inattendue lors de la génération: {e}") # Ajout d'un log pour le débogage serveur
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {type(e).__name__}: {str(e)}")

