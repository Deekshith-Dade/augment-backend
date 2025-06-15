import base64
from openai import OpenAI
from dotenv import load_dotenv
import boto3
import os
from app.utils.aws_utils import get_file_from_s3
load_dotenv()


client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    image_data = get_file_from_s3(image_path)
    return base64.b64encode(image_data).decode("utf-8")

def get_image_description(image_path, full_content):
    base64_image = encode_image(image_path)
    completion =  client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": f"Write a short description of the image, name if you identify someone or any objects from the image try to capture the essence of what the image is aobut. The user uploaded this image with respect to the following context: {full_content}" },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content