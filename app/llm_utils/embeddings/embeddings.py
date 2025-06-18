from openai import AsyncOpenAI
from typing import List

from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI()

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

# def embed_text(text: str) -> list:
#     with torch.no_grad():
#         token = tokenizer(text)
#         embedding = model.encode_text(token).float()
#         return embedding[0].tolist()
    
# def embed_image(image: Image.Image) -> list:
#     with torch.no_grad():
#         image_input = preprocess(image).unsqueeze(0)
#         embedding = model.encode_image(image_input).float()
#         return embedding[0].tolist()

# Embedding using openai text-embedding-3-small
async def embed_text_openai(text: str) -> List:
    
    response = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding