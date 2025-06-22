from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session
from app.models.models import User
from app.database.database import get_db
from svix.webhooks import Webhook, WebhookVerificationError
import os
from app.core.logging import logger
from app.core.limiter import limiter, rate_limits

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

WEBHOOK_SECRET = os.getenv("CLERK_WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    raise RuntimeError("Missing CLERK_WEBHOOK_SIGNING_SECRET in environment")

@router.post("/clerk_auth")
@limiter.limit(rate_limits["RATE_LIMIT_WEBHOOKS_CLERK_AUTH"][0])
async def clerk_auth(request: Request, db: Session = Depends(get_db)):
    logger.info("clerk_auth_request")
    raw_body = await request.body()
    headers = {
        "svix-id": request.headers.get("svix-id", ""),
        "svix-timestamp": request.headers.get("svix-timestamp", ""),
        "svix-signature": request.headers.get("svix-signature", ""),
    }
    
    try:
        verified = Webhook(WEBHOOK_SECRET).verify(raw_body, headers)
    except WebhookVerificationError:
        logger.error("clerk_auth_request_invalid_signature")
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    logger.info("clerk_auth_request_verified")
    
    try:
        data = verified["data"]
        event_type = verified["type"]
        external_id = data.get("id")
        if not external_id:
            raise HTTPException(status_code=400, detail="Missing Clerk user ID")
        logger.info("clerk_auth_request_external_id", external_id=external_id)
        # Process events
        if event_type in ("user.created", "user.updated"):
            email = data.get("email_addresses", [{}])[0].get("email_address", "")
            first = data.get("first_name", "") or ""
            last = data.get("last_name", "") or ""
            name = f"{first} {last}".strip() or email
            profile_image_url = data.get("image_url", "")
            logger.info("clerk_auth_request_user", email=email, first=first, last=last, name=name, profile_image_url=profile_image_url)
            user = db.query(User).filter_by(external_id=external_id).first()
            if not user:
                user = User(
                    external_id=external_id,
                    email=email,
                    name=name,
                    first_name=first,
                    last_name=last,
                    profile_image_url=profile_image_url,
                )
                db.add(user)
            else:
                user.email = email
                user.name = name
                user.first_name = first
                user.last_name = last
                user.profile_image_url = profile_image_url
            db.commit()
            logger.info("clerk_auth_request_return", user_id=user.id)
            return {"message": f"User {event_type} synced successfully"}

        elif event_type == "user.deleted":
            user = db.query(User).filter_by(external_id=external_id).first()
            if user:
                db.delete(user)
                db.commit()
            return {"message": "User deleted successfully"}
        logger.info("clerk_auth_request_return", user_id=user.id)
        return {"message": f"Ignored unsupported event type: {event_type}"}
    except Exception as e:
        logger.error("clerk_auth_request_error", error=str(e), event_type=event_type, external_id=external_id)
        raise HTTPException(status_code=500, detail="Internal server error")
