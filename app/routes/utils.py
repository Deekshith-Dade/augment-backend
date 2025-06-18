import requests
import jwt
import base64
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.models.models import User

from dotenv import load_dotenv

load_dotenv()



auth_scheme = HTTPBearer()

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")

class AuthenticationException(Exception):
    pass

def validate_token(auth_header: str) -> tuple[str, dict]:
    try:
        token = auth_header
    except (AttributeError, IndexError):
        raise AuthenticationException("No authentication token provided")
    jwks = requests.get(CLERK_JWKS_URL, headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"}).json()
    if "keys" not in jwks or not jwks["keys"]:
        raise AuthenticationException("Invalid JWKS")
    key = jwks["keys"][0]
    numbers = RSAPublicNumbers(
        e=int.from_bytes(base64.urlsafe_b64decode(key["e"] + "=" * (-len(key["e"]) % 4)), "big"),
        n=int.from_bytes(base64.urlsafe_b64decode(key["n"] + "=" * (-len(key["n"]) % 4)), "big")
    )
    public_key = numbers.public_key(backend=default_backend())
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    try:
        payload = jwt.decode(
            token,
            pem,
            algorithms=["RS256"],
            options={"verify_signature": True}
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationException("Token has expired.")
    except jwt.DecodeError:
        raise AuthenticationException("Token decode error.")
    except jwt.InvalidTokenError as e:
        raise AuthenticationException(f"Invalid token: {str(e)}")
    
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationException("Token does not contain a user ID")
    
    return user_id, payload

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    db: Session = Depends(get_db)
) -> User:
    try:
        clerk_user_id, payload = validate_token(credentials.credentials)
    except AuthenticationException as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    user = db.query(User).filter(User.external_id == clerk_user_id).first()
    
    if not user:
        # Extract user data from JWT payload
        email = payload.get("email", "")
        first_name = payload.get("first_name", "")
        last_name = payload.get("last_name", "")
        profile_image_url = payload.get("picture", "")
        name = f"{first_name} {last_name}".strip() or email
        
        user = User(
            external_id=clerk_user_id,
            email=email,
            name=name,
            profile_image_url=profile_image_url,
            first_name=first_name,
            last_name=last_name
        )
        db.add(user)
        db.commit()
    
    return user