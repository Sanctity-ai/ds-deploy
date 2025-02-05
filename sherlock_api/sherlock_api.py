from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional
import json
import logging
import os
import asyncio
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv
from database import (
    connect_to_pinecone,
    load_bert_model,
    generate_embedding,
    upsert_questions_and_answers
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

############################################################
# Settings and Models Section
############################################################

class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    domain: str = "venture_capital"
    environment: str = "development"
    user_id: str = "default_user"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

class Meaning(BaseModel):
    literal: str
    inference: str
    subconscious: str

class Response(BaseModel):
    question: str
    answer: str
    meanings: Meaning

class Weightages(BaseModel):
    literal: float
    inference: float
    subconscious: float

class MCQ(BaseModel):
    question: str
    options: Dict[str, str]

class FollowUpRequest(BaseModel):
    responses: List[Response]
    weightages: Optional[Weightages] = None

class FollowUpResponse(BaseModel):
    questions: List[MCQ]

############################################################
# Config Loader Section
############################################################

class ConfigLoader:
    @staticmethod
    def load_json(file_path: str) -> dict:
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=500, detail="Configuration file not found")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            raise HTTPException(status_code=500, detail="Invalid configuration file")

    @staticmethod
    def validate_prompts(prompts: dict) -> None:
        required_keys = {'follow_up_mcq', 'profile_generation'}
        if not all(key in prompts for key in required_keys):
            raise HTTPException(status_code=500, detail="Invalid prompts configuration")

############################################################
# Request Validator Section
############################################################

class RequestValidator:
    @staticmethod
    def validate_responses(responses: List[Response]) -> None:
        if not responses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one response is required"
            )
        
        for idx, response in enumerate(responses):
            if not response.question.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Empty question found at index {idx}"
                )
            if not response.answer.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Empty answer found at index {idx}"
                )

    @staticmethod
    def validate_weightages(weightages: Weightages) -> None:
        total = weightages.literal + weightages.inference + weightages.subconscious
        if not (0.99 <= total <= 1.01):  # Allow for small floating-point errors
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Weightages must sum to 1.0 (current sum: {total})"
            )

############################################################
# GPT Service Section
############################################################

class GPTService:
    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.prompts = ConfigLoader.load_json('prompts.json')
        ConfigLoader.validate_prompts(self.prompts)
        
        # Initialize Pinecone and BERT
        self.pinecone_index = connect_to_pinecone(
            settings.pinecone_api_key,
            settings.pinecone_index_name
        )
        self.tokenizer, self.model = load_bert_model()

    def generate_follow_up_mcq_weighted(
        self,
        initial_question: str,
        user_answer: str,
        meanings: dict,
        weightages: dict
    ) -> Optional[Dict]:
        try:
            weighted_description = (
                f"Literal Meaning ({weightages['literal']}): {meanings['literal']}\n"
                f"Inference Meaning ({weightages['inference']}): {meanings['inference']}\n"
                f"Subconscious Meaning ({weightages['subconscious']}): {meanings['subconscious']}"
            )
            
            prompt = self.prompts['follow_up_mcq'].format(
                initial_question=initial_question,
                user_answer=user_answer,
                weighted_description=weighted_description
            )

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=250,
                n=1
            )
            
            follow_up_mcq = response.choices[0].message.content.strip()
            logger.info(f"Generated MCQ from GPT-4:\n{follow_up_mcq}")
            return self.parse_mcq(follow_up_mcq)
            
        except Exception as e:
            logger.error(f"Error generating follow-up MCQ: {e}")
            return None

    @staticmethod
    def parse_mcq(text: str) -> Optional[Dict]:
        import re
        lines = text.strip().split('\n')
        question_line = None
        options = {}

        for line in lines:
            if not line.strip():
                continue
            if not re.match(r'^[A-D]\)', line.strip()):
                question_line = (question_line or "") + ' ' + line.strip()
            else:
                match = re.match(r'^([A-D])\)\s*(.*)', line.strip())
                if match:
                    options[match.group(1)] = match.group(2)

        if question_line and len(options) == 4:
            return {'question': question_line.strip(), 'options': options}
        return None

############################################################
# Dependencies Section
############################################################

@lru_cache()
def get_gpt_service() -> GPTService:
    return GPTService()

def get_default_weightages() -> Weightages:
    return Weightages(literal=0.5, inference=0.2, subconscious=0.3)

############################################################
# FastAPI App Section
############################################################

app = FastAPI(
    title="VC Survey API",
    description="API for generating follow-up questions based on survey responses",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vc/getFollowUpQuestions", response_model=FollowUpResponse)
async def get_follow_up_questions(
    request: FollowUpRequest,
    gpt_service: GPTService = Depends(get_gpt_service),
    default_weightages: Weightages = Depends(get_default_weightages),
    settings: Settings = Depends(get_settings)
) -> FollowUpResponse:
    """
    Generate follow-up questions based on initial responses.
    
    Args:
        request (FollowUpRequest): Contains initial responses and optional weightages
        gpt_service (GPTService): Service for generating questions
        default_weightages (Weightages): Default weightages if not provided
        settings (Settings): Application settings
    
    Returns:
        FollowUpResponse: Generated follow-up questions
    """
    try:
        # Validate request
        RequestValidator.validate_responses(request.responses)
        weightages = request.weightages or default_weightages
        RequestValidator.validate_weightages(weightages)

        follow_up_questions = []

        # Store initial responses in Pinecone
        initial_questions = [r.question for r in request.responses]
        initial_answers = [r.answer for r in request.responses]
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: upsert_questions_and_answers(
                index=gpt_service.pinecone_index,
                tokenizer=gpt_service.tokenizer,
                model=gpt_service.model,
                questions_list=initial_questions,
                answers_list=initial_answers,
                domain=settings.domain,
                user_id=settings.user_id,
                level=0
            )
        )

        # Generate follow-up questions
        for response in request.responses:
            mcq = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gpt_service.generate_follow_up_mcq_weighted(
                    response.question,
                    response.answer,
                    response.meanings.dict(),
                    weightages.dict()
                )
            )
            
            if mcq:
                follow_up_questions.append(MCQ(**mcq))
            else:
                logger.warning(f"Failed to generate MCQ for question: {response.question}")

        if not follow_up_questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any follow-up questions"
            )

        return FollowUpResponse(questions=follow_up_questions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing follow-up questions request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

############################################################
# Error Handlers Section
############################################################

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status_code": exc.status_code,
        "detail": exc.detail
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "status_code": 500,
        "detail": "Internal server error"
    }

############################################################
# Main Entry Point
############################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)