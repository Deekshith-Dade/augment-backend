from openai import OpenAI
from dotenv import load_dotenv
from app.utils.aws_utils import get_file_from_s3
import io
load_dotenv()



client = OpenAI()

def get_audio_transcript(audio_url):
    audio_file = get_file_from_s3(audio_url)
    buffer = io.BytesIO(audio_file)
    buffer.name = audio_url
    transcript = client.audio.transcriptions.create(
        file=buffer,
        model="whisper-1"
    )
    return transcript.text

if __name__ == "__main__":
    audio_url = "/Users/deekshith/Downloads/test.mp3"
    print(get_audio_transcript(audio_url))