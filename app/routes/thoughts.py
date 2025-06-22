import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.orm import Session

from app.models.models import Thought, Tag, User

from app.schemas.schemas import  ThoughtResponse, VisualizeThought, VisualizeThoughtResponse, ThoughtResponseFull   

from app.database.database import get_db

from app.llm_utils.embeddings.embeddings import embed_text_openai
from app.llm_utils.embeddings.image_utils import get_image_description
from app.llm_utils.embeddings.audio_utils import get_audio_transcript
from app.database.tags import assign_tags_to_thought

from app.utils.aws_utils import upload_file_to_s3, generate_presigned_url
from app.llm_utils.tags import generate_tags_and_title
from app.routes.utils import get_current_user
from app.core.logging import logger
import numpy as np
from sklearn.cluster import KMeans
import umap
import random
import re
from app.core.limiter import limiter, rate_limits

router = APIRouter(prefix="/thoughts", tags=["thoughts"])

@router.post("/create", response_model=ThoughtResponse)
@limiter.limit(rate_limits["RATE_LIMIT_THOUGHTS_CREATE"][0])
async def create_thought(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info("create_thought_request", user_id=user.id)
    thought_id = str(uuid.uuid4())
    user_id = user.id
    file_path = f"user_{user_id}/thoughts/{thought_id}"
    image_url = None
    audio_url = None
    full_content = ""
    
    try:
        if text:
            full_content += f"<Thought>: {text} </Thought>"
        logger.info("create_thought_request_text", user_id=user.id)
        if image:
            image_bytes = await image.read()
            image_url = upload_file_to_s3(f"{file_path}/image.png", image_bytes)
            image_description = get_image_description(f"{file_path}/image.png", full_content)
            full_content += f"\n\n<Image>: {image_description} </Image>"
            logger.info("create_thought_request_image", user_id=user.id, image_url=image_url)
        
        if audio:
            audio_bytes = await audio.read()
            audio_url = upload_file_to_s3(f"{file_path}/audio.mp3", audio_bytes)
            audio_description = get_audio_transcript(f"{file_path}/audio.mp3")
            full_content += f"\n\n<Audio>: {audio_description} </Audio>"
            logger.info("create_thought_request_audio", user_id=user.id, audio_url=audio_url)
        embedding = await embed_text_openai(full_content)
        logger.info("create_thought_request_embedding", user_id=user.id, len_embedding=len(embedding))
        tags = db.query(Tag).filter(Tag.user_id == user_id).all()
        title, tags = generate_tags_and_title(full_content, tags)
        logger.info("create_thought_request_tags", user_id=user.id)
        new_thought = Thought(
            id = thought_id,
            user_id = user_id,
            title = title,
            text_content = text,
            image_url = image_url,
            audio_url = audio_url,
            embedding = embedding,
            full_content = full_content,
            meta = metadata
        )
        logger.info("create_thought_request_new_thought", user_id=user.id)
        db.add(new_thought)
        db.flush()
        
        assign_tags_to_thought(db, user_id, new_thought.id, tags)
        logger.info("create_thought_request_assign_tags", user_id=user.id)
        db.commit()
        db.refresh(new_thought)
        logger.info("create_thought_request_commit", user_id=user.id)
    except Exception as e:
        logger.error("create_thought_request_error", user_id=user.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("create_thought_request_return", user_id=user.id, thought_id=thought_id)
    return ThoughtResponse(id=thought_id, created_at=str(new_thought.created_at))


@router.get("/visualize", response_model=VisualizeThoughtResponse)
@limiter.limit(rate_limits["RATE_LIMIT_THOUGHTS_READ"][0])
async def get_clustered_thoughts(
    request: Request,
    db: Session = Depends(get_db), 
    n_components: int = 3, 
    n_clusters: int = 5, 
    user: User = Depends(get_current_user)
):
    logger.info("get_clustered_thoughts_request",  user_id=user.id)
    user_id = user.id
    try:
        thoughts = db.query(Thought).filter(Thought.user_id == user_id).all()
        logger.info("get_clustered_thoughts_request_thoughts", user_id=user.id, len_thoughts=len(thoughts))
        if not thoughts:
            logger.error("get_clustered_thoughts_request_no_thoughts", user_id=user.id)
            raise HTTPException(status_code=404, detail="No thoughts found")
        
        if len(thoughts) <  5:
            logger.info("get_clustered_thoughts_request_less_than_5_thoughts", user_id=user.id)
            response = []
            for i, thought in enumerate(thoughts):
                response.append(VisualizeThought(
                    id = str(thought.id),
                    title = thought.title,
                    excerpt = thought.text_content,
                    created_at = str(thought.created_at),
                    position = [random.random() * 2, random.random() * 2, random.random() * 2],
                    label = i
                ))
            logger.info("get_clustered_thoughts_request_return", user_id=user.id)
            return VisualizeThoughtResponse(thoughts=response)
        
        embeddings = np.array([t.embedding for t in thoughts])
        
        # UMAP Dimensionality Reduction
        desired_n_neighbors = max(10, int(len(embeddings) * 0.05))
        n_neighbors = min(len(embeddings) - 1, desired_n_neighbors)
        logger.info("get_clustered_thoughts_request_n_neighbors", user_id=user.id, n_neighbors=n_neighbors)
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, n_jobs=1)
        reduced = reducer.fit_transform(embeddings)
        
        kemans = KMeans(n_clusters=n_clusters)
        labels = kemans.fit_predict(reduced)
        logger.info("get_clustered_thoughts_request_labels", user_id=user.id)
        # Create response
        response = []
        for i, thought in enumerate(thoughts):
            response.append(VisualizeThought(
                id = str(thought.id),
                title = thought.title,
                excerpt = thought.text_content,
                created_at = str(thought.created_at),
                position = reduced[i].tolist(),
                label = labels[i]
            ))
    except Exception as e:
        logger.error("get_clustered_thoughts_request_error", user_id=user.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
            
    logger.info("get_clustered_thoughts_request_return", user_id=user.id)
    return VisualizeThoughtResponse(thoughts=response)
    
@router.get("/{thought_id}", response_model=ThoughtResponseFull)
@limiter.limit(rate_limits["RATE_LIMIT_THOUGHTS_READ"][0])
async def get_thought(request: Request, thought_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    logger.info("get_thought_request", user_id=user.id, thought_id=thought_id)
    user_id = user.id
    thought = db.query(Thought).filter(Thought.id == thought_id, Thought.user_id == user_id).first()
    logger.info("get_thought_request_thought", user_id=user.id)
    if not thought:
        logger.error("get_thought_request_thought_not_found", user_id=user.id, thought_id=thought_id)
        raise HTTPException(status_code=404, detail="Thought not found")

    logger.info("get_thought_request_return", user_id=user.id)
    return ThoughtResponseFull(
        id = str(thought.id),
        title = thought.title,
        text_content = thought.text_content,
        image_url = generate_presigned_url(thought.image_url) if thought.image_url else None,
        audio_url = generate_presigned_url(thought.audio_url) if thought.audio_url else None,
        full_content = thought.full_content,
        created_at = str(thought.created_at),
        updated_at = str(thought.updated_at)
    )
    
@router.put("/{thought_id}", response_model=ThoughtResponse)
@limiter.limit(rate_limits["RATE_LIMIT_THOUGHTS_UPDATE"][0])
async def update_thought(
    request: Request,
    thought_id: str,
    title: str = Form(...),
    text_content: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info("update_thought_request", user_id=user.id, thought_id=thought_id)
    user_id = user.id
    thought = db.query(Thought).filter(Thought.id == thought_id, Thought.user_id == user_id).first()

    if not thought:
        logger.error("update_thought_request_thought_not_found", user_id=user.id, thought_id=thought_id)
        raise HTTPException(status_code=404, detail="Thought not found")

    try:
        logger.info("update_thought_request_thought_found", user_id=user.id, thought_id=thought_id)
        file_path = f"user_{user_id}/thoughts/{thought_id}"
        pre_full_content = thought.full_content or ""

        # Replace or add <Thought>
        if re.search(r"<Thought>.*?</Thought>", pre_full_content, re.DOTALL):
            full_content = re.sub(r"<Thought>.*?</Thought>", f"<Thought>{text_content}</Thought>", pre_full_content, flags=re.DOTALL)
        else:
            full_content = pre_full_content + f"\n\n<Thought>{text_content}</Thought>"

        # Handle image
        if image:
            image_bytes = await image.read()
            image_url = upload_file_to_s3(f"{file_path}/image.png", image_bytes)
            image_description = get_image_description(f"{file_path}/image.png", full_content)

            if re.search(r"<Image>.*?</Image>", full_content, re.DOTALL):
                full_content = re.sub(r"<Image>.*?</Image>", f"<Image>{image_description}</Image>", full_content, flags=re.DOTALL)
            else:
                full_content += f"\n\n<Image>{image_description}</Image>"

            thought.image_url = image_url

        # Handle audio
        if audio:
            audio_bytes = await audio.read()
            audio_url = upload_file_to_s3(f"{file_path}/audio.mp3", audio_bytes)
            audio_description = get_audio_transcript(f"{file_path}/audio.mp3")

            if re.search(r"<Audio>.*?</Audio>", full_content, re.DOTALL):
                full_content = re.sub(r"<Audio>.*?</Audio>", f"<Audio>{audio_description}</Audio>", full_content, flags=re.DOTALL)
            else:
                full_content += f"\n\n<Audio>{audio_description}</Audio>"

            thought.audio_url = audio_url

        # Update thought
        thought.title = title
        thought.text_content = text_content
        thought.full_content = full_content
        thought.embedding = await embed_text_openai(full_content)

        # Tag generation and assignment
        tags = db.query(Tag).filter(Tag.user_id == user_id).all()
        _, tags = generate_tags_and_title(full_content, tags)
        assign_tags_to_thought(db, user_id, thought.id, tags)

        db.commit()
        db.refresh(thought)
        logger.info("update_thought_request_return", user_id=user.id, thought_id=thought_id)
        return ThoughtResponse(id=str(thought.id), created_at=str(thought.created_at))

    except Exception as e:
        logger.error("update_thought_request_error", user_id=user.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{thought_id}")
@limiter.limit(rate_limits["RATE_LIMIT_THOUGHTS_DELETE"][0])
async def delete_thought(request: Request, thought_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    logger.info("delete_thought_request", user_id=user.id, thought_id=thought_id)
    try:
        user_id = user.id
        thought = db.query(Thought).filter(Thought.id == thought_id, Thought.user_id == user_id).first()
        logger.info("delete_thought_request_thought", user_id=user.id)
        if not thought:
            logger.error("delete_thought_request_thought_not_found", user_id=user.id, thought_id=thought_id)
            raise HTTPException(status_code=404, detail="Thought not found")
            
        db.delete(thought)
        db.commit()
        logger.info("delete_thought_request_return", user_id=user.id, thought_id=thought_id)
        return {"message": "Thought deleted successfully"}
    except Exception as e:
        logger.error("delete_thought_request_error", user_id=user.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))