from fastapi import APIRouter, HTTPException, Request, Depends
from app.models.models import User
from app.database.database import get_db
from sqlalchemy.orm import Session
import hmac
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
CLERK_WEBHOOK_SECRET = os.getenv("CLERK_WEBHOOK_SECRET")
print("secret", CLERK_WEBHOOK_SECRET)

def verify_clerk_webhook(request: Request, raw_body: bytes):
    signature = request.headers.get("clerk-signature")
    
    if not signature:
        raise HTTPException(status_code=400, detail="Missing clerk-signature")
    
    try:
        expected_signature = hmac.new(
            CLERK_WEBHOOK_SECRET.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise HTTPException(status_code=400, detail="Invalid clerk-signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/clerk_auth")
async def clerk_auth(request: Request, db: Session = Depends(get_db)):
    raw_body = await request.body()
    
    # verify_clerk_webhook_signature(request, raw_body)
    
    body = await request.json()
    
    event_type = body.get("type")
    data = body.get("data", {})
    external_id = data.get("id")

    if not external_id:
        raise HTTPException(status_code=400, detail="Missing Clerk user ID")

    if event_type in ("user.created", "user.updated"):
        email = data.get("email_addresses", [{}])[0].get("email_address")
        first_name = data.get("first_name") or ""
        last_name = data.get("last_name") or ""
        profile_image_url = data.get("image_url") or ""
        name = f"{first_name} {last_name}".strip() or email

        user = db.query(User).filter_by(external_id=external_id).first()
        if user:
            # Update existing user
            user.email = email
            user.name = name
            user.profile_image_url = profile_image_url
            user.first_name = first_name
            user.last_name = last_name
        else:
            # Create new user
            user = User(
                external_id=external_id,
                email=email,
                name=name,
                profile_image_url=profile_image_url,
                first_name=first_name,
                last_name=last_name
            )
            db.add(user)

        db.commit()
        return {"message": f"User {event_type.replace('user.', '')} successfully"}

    elif event_type == "user.deleted":
        user = db.query(User).filter_by(external_id=external_id).first()
        if user:
            db.delete(user)
            db.commit()
        return {"message": "User deleted successfully"}

    return {"message": f"Unhandled event type: {event_type}"}
