from fastapi import APIRouter
from api.schemas.input import FeedbackInput
from api.utils.db import save_feedback

router = APIRouter()

@router.post("/")
async def submit_feedback(feedback: FeedbackInput):
    try:
        save_feedback(feedback.text, feedback.label)
        return {"message": "Feedback saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
