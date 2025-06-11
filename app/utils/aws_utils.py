import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def upload_file_to_s3(filename: str, file_bytes: bytes) -> str:
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=filename, Body=file_bytes)
        return f"https://{BUCKET_NAME}.s3.amazonaws.com/{filename}"
    except (BotoCoreError, NoCredentialsError) as e:
        logger.exception("S3 upload failed")
        raise RuntimeError("Failed to upload to S3") from e

def get_file_from_s3(filepath: str) -> bytes:
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=filepath)
        return response['Body'].read()
    except (BotoCoreError, NoCredentialsError) as e:
        logger.exception("S3 download failed")
        raise RuntimeError("Failed to download from S3") from e
