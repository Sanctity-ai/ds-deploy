import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import re
import time
from openai import OpenAI
from pathlib import Path
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Question Generator API")

# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

# Pydantic models
class MCQOption(BaseModel):
    A: str
    B: str
    C: str
    D: str

class MCQResponse(BaseModel):
    question: str
    options: MCQOption

class MCQInternal(MCQResponse):
    embedding: List[float]

class ProblemStatement(BaseModel):
    problem_statement: str
    funding_stage: str
    main_goal: str
    preferred_industry: str

# Database functions
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
        index = pc.Index(PINECONE_INDEX_NAME)
    
    return index

def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def generate_embedding(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()

def store_in_pinecone(index, mcq: MCQInternal, metadata: dict):
    try:
        unique_id = f"mcq_{datetime.now().timestamp()}"
        
        options_dict = mcq.options.dict()
        metadata.update({
            "question": mcq.question,
            "option_a": options_dict["A"],
            "option_b": options_dict["B"],
            "option_c": options_dict["C"],
            "option_d": options_dict["D"],
            "timestamp": datetime.now().isoformat()
        })
        
        index.upsert([
            {
                "id": unique_id,
                "values": mcq.embedding,
                "metadata": metadata
            }
        ])
        return unique_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store in database: {str(e)}")

# Load prompts
def load_prompts():
    prompts_path = Path("prompts.json")
    if not prompts_path.exists():
        raise FileNotFoundError("prompts.json file not found")
    
    with open(prompts_path, "r") as f:
        return json.load(f)

PROMPTS = load_prompts()

# Configure OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize BERT and Pinecone at startup
tokenizer, model = load_bert_model()
pinecone_index = initialize_pinecone()

class QuestionGenerator:
    def __init__(self, problem_statement: str, funding_stage: str, main_goal: str, preferred_industry: str):
        self.problem_statement = problem_statement
        self.funding_stage = funding_stage
        self.main_goal = main_goal
        self.preferred_industry = preferred_industry
        self.model_name = "gpt-4"
        self.client = client

    def generate_mcqs(self, prompt: str, retries: int = 3, delay: int = 2) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,
                    n=1,
                    temperature=0.3,
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(max(1, delay - attempt))
                    continue
                raise HTTPException(status_code=500, detail=f"Failed to generate MCQs: {str(e)}")

    def parse_mcqs(self, raw_text: str) -> List[dict]:
        if not raw_text:
            raise HTTPException(status_code=400, detail="No text to parse")

        try:
            lines = raw_text.strip().split('\n')
            current_question = None
            options = {}
            mcqs = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if not line.startswith(('A)', 'B)', 'C)', 'D)')):
                    if current_question and len(options) == 4:
                        mcqs.append({
                            "question": current_question,
                            "options": options
                        })
                        options = {}
                    current_question = line
                else:
                    option_letter = line[0]
                    option_text = line[2:].strip()
                    options[option_letter] = option_text
            
            if current_question and len(options) == 4:
                mcqs.append({
                    "question": current_question,
                    "options": options
                })

            return mcqs[:20]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing MCQs: {str(e)}")

@app.post("/vc/getAiGeneratedQuestions", response_model=List[MCQResponse])
async def get_ai_generated_questions(problem: ProblemStatement) -> List[MCQResponse]:
    """
    Generate AI-powered multiple choice questions and store them in Pinecone.
    """
    question_generator = QuestionGenerator(
        problem.problem_statement,
        problem.funding_stage,
        problem.main_goal,
        problem.preferred_industry
    )
    
    prompt = PROMPTS["initial_question_prompt"].format(
        problem_statement=problem.problem_statement,
        funding_stage=problem.funding_stage,
        main_goal=problem.main_goal,
        preferred_industry=problem.preferred_industry
    )
    
    raw_mcqs = question_generator.generate_mcqs(prompt)
    if not raw_mcqs:
        raise HTTPException(status_code=500, detail="Failed to generate questions")
    
    parsed_mcqs = question_generator.parse_mcqs(raw_mcqs)
    if not parsed_mcqs:
        raise HTTPException(status_code=500, detail="Failed to parse questions")
    
    mcq_objects = []
    metadata = {
        "problem_statement": problem.problem_statement,
        "funding_stage": problem.funding_stage,
        "main_goal": problem.main_goal,
        "preferred_industry": problem.preferred_industry
    }
    
    for mcq_dict in parsed_mcqs:
        # Create internal MCQ object with embedding
        mcq_internal = MCQInternal(
            question=mcq_dict["question"],
            options=MCQOption(**mcq_dict["options"]),
            embedding=generate_embedding(mcq_dict["question"], tokenizer, model)
        )
        
        # Store in Pinecone
        store_in_pinecone(pinecone_index, mcq_internal, metadata.copy())
        
        # Create response MCQ object without embedding
        mcq_response = MCQResponse(
            question=mcq_dict["question"],
            options=MCQOption(**mcq_dict["options"])
        )
        
        mcq_objects.append(mcq_response)
    
    return mcq_objects

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)