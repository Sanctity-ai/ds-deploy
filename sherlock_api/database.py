import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import logging
from typing import Tuple, List, Optional, Dict
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class ModelInitError(Exception):
    """Custom exception for model initialization errors"""
    pass

class PineconeConnectionError(Exception):
    """Custom exception for Pinecone connection errors"""
    pass

class EmbeddingGenerationError(Exception):
    """Custom exception for embedding generation errors"""
    pass

# Constants
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

@lru_cache(maxsize=1)
def get_device() -> torch.device:
    """
    Get PyTorch device with caching.
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_pinecone(
    api_key: str,
    index_name: str,
    dimension: int = 768,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-west-2"
) -> Pinecone:
    """Initialize a new Pinecone index with serverless specification."""
    try:
        pc = Pinecone(api_key=api_key)
        
        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            logger.info(f"Created new Pinecone index: {index_name}")
        
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise PineconeConnectionError(f"Failed to initialize Pinecone: {str(e)}")

@lru_cache(maxsize=1)
def connect_to_pinecone(api_key: str, index_name: str) -> Pinecone:
    """Connect to Pinecone with connection caching."""
    try:
        pc = Pinecone(api_key=api_key)
        
        if index_name not in pc.list_indexes().names():
            raise ValueError(f"Index '{index_name}' does not exist")
            
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Pinecone connection failed: {str(e)}")
        raise PineconeConnectionError(f"Failed to connect to Pinecone: {str(e)}")

@lru_cache(maxsize=1)
def load_bert_model() -> Tuple[BertTokenizer, BertModel]:
    """Load BERT model with caching."""
    try:
        device = get_device()
        logger.info(f"Using device: {device}")
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load BERT model: {e}")
        raise ModelInitError(f"Failed to load BERT model: {str(e)}")

def generate_embedding(
    text: str,
    tokenizer: BertTokenizer,
    model: BertModel,
    retry_count: int = 0
) -> List[float]:
    """Generate embeddings with retry mechanism."""
    try:
        device = get_device()
        
        tokens = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
            
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings[0].cpu().tolist()
    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Retry {retry_count + 1} for embedding generation")
            asyncio.sleep(RETRY_DELAY)
            return generate_embedding(text, tokenizer, model, retry_count + 1)
        logger.error(f"Failed to generate embedding: {e}")
        raise EmbeddingGenerationError(f"Failed to generate embedding: {str(e)}")

def create_upsert_payload(
    text: str,
    level: int,
    type: str,
    domain: str,
    user_id: str,
    embedding: List[float],
    associated_id: Optional[str] = None
) -> Dict:
    """Create payload for Pinecone upsert."""
    unique_id = f"{type}_level{level}_{datetime.now().timestamp()}"
    metadata = {
        "type": f"level{level}_{type}",
        "text": text,
        "domain": domain,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
    }
    
    if associated_id:
        metadata["associated_id"] = associated_id

    return {
        "id": unique_id,
        "values": embedding,
        "metadata": metadata
    }

async def batch_upsert_to_pinecone(
    index: Pinecone,
    vectors: List[Dict],
    retry_count: int = 0
) -> None:
    """Batch upsert vectors to Pinecone with retry mechanism."""
    try:
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            index.upsert(batch)
            logger.info(f"Upserted batch {i//BATCH_SIZE + 1}/{(len(vectors) + BATCH_SIZE - 1)//BATCH_SIZE}")
    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Retry {retry_count + 1} for batch upsert")
            await asyncio.sleep(RETRY_DELAY)
            await batch_upsert_to_pinecone(index, vectors, retry_count + 1)
        else:
            logger.error(f"Failed to upsert batch to Pinecone: {e}")
            raise

async def upsert_questions_and_answers(
    index: Pinecone,
    tokenizer: BertTokenizer,
    model: BertModel,
    questions_list: List[str],
    answers_list: List[str],
    domain: str,
    user_id: str,
    level: int
) -> None:
    """Asynchronously upsert questions and answers with batching."""
    try:
        if len(questions_list) != len(answers_list):
            raise ValueError("Questions and answers lists must have same length")

        vectors = []
        
        with ThreadPoolExecutor() as executor:
            # Generate embeddings in parallel
            question_embeddings = list(executor.map(
                lambda q: generate_embedding(q, tokenizer, model),
                questions_list
            ))
            answer_embeddings = list(executor.map(
                lambda a: generate_embedding(a, tokenizer, model),
                answers_list
            ))

        # Create payloads for questions and answers
        for idx, (question, answer, q_emb, a_emb) in enumerate(zip(
            questions_list, answers_list, question_embeddings, answer_embeddings
        )):
            # Create question payload
            question_payload = create_upsert_payload(
                text=question,
                level=level,
                type="question",
                domain=domain,
                user_id=user_id,
                embedding=q_emb
            )
            vectors.append(question_payload)
            
            # Create answer payload
            answer_payload = create_upsert_payload(
                text=answer,
                level=level,
                type="answer",
                domain=domain,
                user_id=user_id,
                embedding=a_emb,
                associated_id=question_payload["id"]
            )
            vectors.append(answer_payload)

        # Batch upsert all vectors
        await batch_upsert_to_pinecone(index, vectors)
        logger.info(f"Successfully processed {len(questions_list)} QA pairs")
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to upsert questions and answers: {e}")
        raise