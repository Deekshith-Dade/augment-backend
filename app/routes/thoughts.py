import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
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

import numpy as np
from sklearn.cluster import KMeans
import umap
import random
import re

router = APIRouter(prefix="/thoughts", tags=["thoughts"])

@router.post("/create", response_model=ThoughtResponse)
async def create_thought(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    thought_id = str(uuid.uuid4())
    user_id = user.id
    file_path = f"user_{user_id}/thoughts/{thought_id}"
    image_url = None
    audio_url = None
    full_content = ""
    
    try:
        if text:
            full_content += f"<Thought>: {text} </Thought>"
        
        if image:
            image_bytes = await image.read()
            image_url = upload_file_to_s3(f"{file_path}/image.png", image_bytes)
            image_description = get_image_description(f"{file_path}/image.png", full_content)
            full_content += f"\n\n<Image>: {image_description} </Image>"
        
        if audio:
            audio_bytes = await audio.read()
            audio_url = upload_file_to_s3(f"{file_path}/audio.mp3", audio_bytes)
            audio_description = get_audio_transcript(f"{file_path}/audio.mp3")
            full_content += f"\n\n<Audio>: {audio_description} </Audio>"
            
        embedding = await embed_text_openai(full_content)
        
        tags = db.query(Tag).filter(Tag.user_id == user_id).all()
        title, tags = generate_tags_and_title(full_content, tags)
        
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
        
        db.add(new_thought)
        db.flush()
        
        assign_tags_to_thought(db, user_id, new_thought.id, tags)
        
        db.commit()
        db.refresh(new_thought)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return ThoughtResponse(id=thought_id, created_at=str(new_thought.created_at))


@router.get("/visualize", response_model=VisualizeThoughtResponse)
async def get_clustered_thoughts(db: Session = Depends(get_db), n_components: int = 3, n_clusters: int = 5, user: User = Depends(get_current_user)):
    user_id = user.id
    
    thoughts = db.query(Thought).filter(Thought.user_id == user_id).all()
    
    if not thoughts:
        raise HTTPException(status_code=404, detail="No thoughts found")
    
    if len(thoughts) <  5:
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
        return VisualizeThoughtResponse(thoughts=response)
    
    embeddings = np.array([t.embedding for t in thoughts])
    
    # UMAP Dimensionality Reduction
    desired_n_neighbors = max(10, int(len(embeddings) * 0.05))
    n_neighbors = min(len(embeddings) - 1, desired_n_neighbors)
    print(f"n_neighbors: {n_neighbors}")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, n_jobs=1)
    reduced = reducer.fit_transform(embeddings)
    
    kemans = KMeans(n_clusters=n_clusters)
    labels = kemans.fit_predict(reduced)
    
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
    
    return VisualizeThoughtResponse(thoughts=response)
    
@router.get("/{thought_id}", response_model=ThoughtResponseFull)
async def get_thought(thought_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    user_id = user.id
    thought = db.query(Thought).filter(Thought.id == thought_id, Thought.user_id == user_id).first()
    
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")
    
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
async def update_thought(
    thought_id: str,
    title: str = Form(...),
    text_content: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    user_id = user.id
    thought = db.query(Thought).filter(Thought.id == thought_id, Thought.user_id == user_id).first()

    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")

    try:
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

        return ThoughtResponse(id=str(thought.id), created_at=str(thought.created_at))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.delete("/{thought_id}")
async def delete_thought(thought_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    user_id = user.id
    thought = db.query(Thought).filter(Thought.id == thought_id, Thought.user_id == user_id).first()
    
    if not thought:
        raise HTTPException(status_code=404, detail="Thought not found")
        
    db.delete(thought)
    db.commit()
    return {"message": "Thought deleted successfully"}