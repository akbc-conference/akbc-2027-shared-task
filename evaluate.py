"""
AKBC 2027 shared task: evaluate a submission.

    python evaluate.py SUBMISSION.csv --split dev --out OUT_DIR

One run does everything, in two stages, and prints precision, recall and F1:

CHECKING (the rules every submission goes through)
  1. read the CSV: columns subject, predicate, object (others ignored). An unreadable file or missing
     columns rejects the submission (exit code 2).
  2. drop rows with an empty field, and rows whose subject is not an entity of the split. Subjects are
     compared after normalising Unicode (NFKC), dashes, spaces and case; kept rows get the exact title.
  3. drop exact duplicates within an entity (same predicate and object after the same normalising).
  4. keep each entity's first 500 triples, in file order.
  5. drop triples that state the same fact as an earlier triple of the entity in other words: one call
     per entity to GLM-5.3-Flash with all the entity's triples (prompt DEDUP_PROMPT below); the model
     lists the repetitions, each with a reason. Triples are kept when a call keeps failing; the run is
     then marked incomplete.
  6. write OUT_DIR/elicited_triples.csv (the checked triples), OUT_DIR/validation_report.json (every
     dropped row with its reason, counts, settings) and OUT_DIR/dedup_calls.jsonl (every model answer).

SCORING (judge: Gemma 4 26B-A4B, temperature 0)
  precision  1,000 of the checked triples are sampled (a random entity, then a random triple of it;
             seed 42). For each, the 10 passages of the entity's reference documents (Wikipedia article
             and web documents) most similar to the triple are retrieved (embeddings
             text-embedding-3-small), and the judge decides whether they entail the triple.
             precision = share entailed.
  recall     1,000 reference triples are sampled the same way (same random generator, after the
             precision sample). For each, the judge sees all of the submission's triples about that
             entity and decides whether they entail it. recall = share entailed.
  F1         harmonic mean of precision and recall.
  Results in OUT_DIR/scores/: scores.json, results.csv, results_detailed.csv (every judged triple with
  the judge's answer and the passages it saw), judge_calls.jsonl.

Stopping is safe at any point: running the same command again keeps every answer already received.

Data: DATA_DIR/<split>/entities.json and DATA_DIR/<split>/reference_corpus.json.xz (default: ./data).
Precision retrieval uses passage embeddings cached in DATA_DIR/<split>/rag_cache/, built on first use
(one embedding call per passage; about 500 entities' documents per split).

Services (OpenAI-compatible APIs), set as environment variables:
  API_BASE_URL, API_KEY    serving zai-org/GLM-5.3-Flash (checking) and google/gemma-4-26B-A4B-it
                           (judge); the organisers use the SCADS API (SCADSAI_BASE_URL and
                           SCADSAI_API_KEY are read too)
  OPENROUTER_API_KEY       embeddings (openai/text-embedding-3-small), only to build passage indexes

The scoring is the scoring path of the BeQu pipeline (https://bequ.scads.ai/), reduced to what the
shared task uses: judge prompts, sampling and retrieval are unchanged, and the organisers check that it
sends the judge exactly the same requests.
"""
import argparse
import csv
import fcntl
import functools
import glob
import gzip
import hashlib
import json
import lzma
import os
import pickle
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import openai
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def api_base_url():
    return os.environ.get("API_BASE_URL") or os.environ.get("SCADSAI_BASE_URL")


def api_key():
    return os.environ.get("API_KEY") or os.environ.get("SCADSAI_API_KEY")


# ===== network =====
class RateLimiter:
    """Spaces out calls so that at most `rpm` of them start per minute, across all threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.interval
        time.sleep(max(0.0, start - now))


# One limiter per process for the judge, shared by every evaluation job and worker thread
_judge_rate_limiter = None


def set_judge_rate_limit(rpm):
    """Limit judge calls to `rpm` per minute in this process (None or 0: no limit)."""
    global _judge_rate_limiter
    _judge_rate_limiter = RateLimiter(rpm) if rpm else None
    if rpm:
        logger.info(f"Judge rate limit: {rpm} requests per minute")


def wait_for_judge_rate_limit():
    if _judge_rate_limiter is not None:
        _judge_rate_limiter.wait()


def network_retry(
    max_retries: int = 10,
    initial_delay: float = 1.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    retry_on_exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying network operations with exponential backoff and jitter.
    
    Args:
        max_retries (int): Maximum number of retry attempts (default: 6)
        initial_delay (float): Initial delay in seconds (default: 1.0)
        max_delay (float): Maximum delay in seconds (default: 120.0)
        exponential_base (float): Base for exponential backoff (default: 2.0)
        retry_on_exceptions: Tuple of exceptions to catch and retry (default: all)
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Failed after {max_retries} retries for {func.__name__}: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                    # Add jitter to avoid synchronized retries
                    delay += random.uniform(0, 0.1 * delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# ===== judge =====
def _parse_json(text: str) -> dict:
    """Parse JSON from LLM output that may contain extra text or markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Repair common truncation: model omits the leading '{' so response starts with '"key":' or 'key":'
    if text and text[0] != '{' and text[0] != '[':
        text = '{' + text
        # Ensure it's closed
        if not text.rstrip().endswith('}'):
            text = text.rstrip().rstrip(',') + '}'

    try:
        result = json.loads(text)
        # If model returned a list, try to find a dict inside it
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    return item
            raise ValueError(f"JSON array with no dict element: {result}")
        return result
    except json.JSONDecodeError:
        # Find the outermost {...} block (supports nested structures)
        depth, start = 0, -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    start = -1
        # Last resort: bare answer letter
        bare = re.search(r'\b([abc])\b', text.lower())
        if bare:
            return {"answer": bare.group(1), "reasoning": "extracted from plain text"}
        raise


class Request:
    """Judge client for one judge model (default route: SCADS via SCADSAI_BASE_URL / SCADSAI_API_KEY)."""

    def __init__(self, llm_judge, judge_api_url: str = None, judge_api_key: str = None):
        """
        Args:
            llm_judge: judge model name
            judge_api_url, judge_api_key: an OpenAI-compatible endpoint; default SCADS
                (environment variables SCADSAI_BASE_URL, SCADSAI_API_KEY)
        """
        self.llm_judge = llm_judge
        # optional callable(kind, triple, response, seconds), called after every judge call
        self.call_log = None
        self.client = OpenAI(base_url=judge_api_url or api_base_url(), api_key=judge_api_key or api_key())

    @network_retry(max_retries=10, initial_delay=1.0)
    def verify_triple_lm_wikidata(self, triple, gold_triples):
        """
        Verify an RDF triple against a list of Wikidata triples.
        
        Used for recall computation (checking if GT triples are covered by elicited triples).
        
        Args:
            triple: RDF triple string in format "(subject, predicate, object)"
            gold_triples: List of gold RDF triples to verify against
            
        Returns:
            dict: {"answer": "a" | "b" | "c", "reasoning": "..."}
                  a=entailment, b=contradiction, c=neutral
        """

        triple_prompt_str = f"Statement to verify: {triple}."
        gold_prompt_str = f"List of triples to verify from: {str(gold_triples)}"
        
        # Define the JSON schema for structured output
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleVerification",
                "description": "Verification of an RDF triple against a list of triples",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["a", "b", "c"],
                            "description": "a) entailment, b) contradiction, c) neutral"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the choice"
                        }
                    },
                    "required": ["answer", "reasoning"]
                }
            }
        }
        
        messages = [
            {"role": "user",
                "content": "Can the given RDF triple be inferred from the given list of triples? \
                            a) The list of triples entails the given RDF triple.\
                            b) The list of triples contradicts the given RDF triple. \
                            c) The truth of the given RDF triple cannot be determined from the list of triples alone. \
                            If the provided evidence is internally CONFLICTING (i.e., it contains information that both supports AND contradicts the statement), you MUST answer 'c' (neutral), rather than picking a side. \
                            Respond with JSON containing 'answer' (one of a/b/c) and 'reasoning'."},
            {"role": "user", "content": triple_prompt_str},
            {"role": "user", "content": gold_prompt_str},
        ]

        logger.debug("=== LLM CALL: verify_triple_lm_wikidata ===")
        logger.debug(f"Triple to verify: {triple}")
        logger.debug(f"Gold triples length: {len(str(gold_triples))} characters")
        logger.debug(f"Model: {self.llm_judge}")

        wait_for_judge_rate_limit()
        started = time.time()
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_judge,
            max_tokens=3000,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            response_format=response_schema,
        )
        if self.call_log:
            self.call_log("recall", triple, response, time.time() - started)

        response_text = response.choices[0].message.content
        result = _parse_json(response_text)

        # Validate structured output
        if not isinstance(result, dict) or result.get("answer") not in ['a', 'b', 'c']:
            raise ValueError(f"Invalid LLM response for triple '{triple}': {result}")

        logger.debug(f"Wikidata verification completed: {triple[:50]}... => {result.get('answer')}")
        
        return result

    @network_retry(max_retries=10, initial_delay=1.0)
    def verify_triple_with_rag(self, triple: str, rag_context: str, sources: list) -> dict:
        """
        Verify an RDF triple against retrieved RAG passages in a single LLM call.
        
        Used for precision evaluation with the RAG-based approach. The LLM receives
        the triple and top-k retrieved passages, then determines if the triple is
        entailed, contradicted, or neutral based on the passages.
        
        Args:
            triple: RDF triple string in format "(subject, predicate, object)"
            rag_context: Combined text of all retrieved passages
            sources: List of source names for retrieved passages (e.g., ['wikipedia', 'wikidata'])
            
        Returns:
            dict: {"answer": "a" | "b" | "c", "reasoning": "..."}
                  a=entailment, b=contradiction, c=neutral
        """
        triple_prompt_str = f"Statement to verify: {triple}."
        context_prompt_str = f"Retrieved ground truth passages:\n{rag_context}"
        
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleVerification",
                "description": "Verification of an RDF triple against retrieved passages",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["a", "b", "c"],
                            "description": "a) entailment, b) contradiction, c) neutral"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the choice"
                        }
                    },
                    "required": ["answer", "reasoning"]
                }
            }
        }
        
        sources_str = ", ".join(sources) if sources else "unknown"
        
        messages = [
            {"role": "user",
                "content": "Can the given RDF triple be inferred from the given retrieved passages? \
                            a) The passages entail the RDF triple.\
                            b) The passages contradict the RDF triple. \
                            c) The truth of the given RDF triple cannot be determined from the passages alone. \
                            If the provided evidence is internally CONFLICTING (i.e., it contains information that both supports AND contradicts the statement), you MUST answer 'c' (neutral), rather than picking a side. \
                            Respond with JSON containing 'answer' (one of a/b/c) and 'reasoning'."},
            {"role": "user", "content": triple_prompt_str},
            {"role": "user", "content": context_prompt_str},
        ]
        
        logger.debug("=== LLM CALL: verify_triple_with_rag ===")
        logger.debug(f"Triple: {triple}")
        logger.debug(f"Sources: {sources_str}")
        logger.debug(f"Model: {self.llm_judge}")

        wait_for_judge_rate_limit()
        started = time.time()
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_judge,
            max_tokens=3000,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            response_format=response_schema,
        )
        if self.call_log:
            self.call_log("precision", triple, response, time.time() - started)

        response_text = response.choices[0].message.content
        result = _parse_json(response_text)

        # Validate structured output
        if not isinstance(result, dict) or result.get("answer") not in ['a', 'b', 'c']:
            raise ValueError(f"Invalid LLM response for triple '{triple}': {result}")

        logger.debug(f"RAG verification completed: {triple[:50]}... => {result.get('answer')}")
        
        return result


# ===== retrieval =====
class GroundTruthRAG:
    """
    Dense (embedding-based) RAG system for ground truth retrieval.
    
    For a given entity, builds an index from:
    - Wikipedia article paragraphs (original_content)
    - All web document paragraphs (from full web documents)
    
    Retrieves top-k passages using cosine similarity between query embedding
    and passage embeddings.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the RAG system.

        Args:
            embedding_model: Embedding model to use, routed through OpenRouter
                (default: text-embedding-3-small, auto-prefixed with "openai/")
        """
        self.embedding_model = embedding_model if "/" in embedding_model else f"openai/{embedding_model}"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.passages = []
        self.embeddings = None
        
    def build_index(self, wiki_data: Dict) -> None:
        """
        Build passage index from ground truth data.
        
        Creates passages from:
        - Wikipedia article: each paragraph becomes a passage
        - Web documents: each paragraph becomes a passage with title prefix
        
        Args:
            wiki_data: Dictionary containing ground truth with keys:
                - original_content: full Wikipedia article text
                - web_full_documents: list of web document dicts with 'content' and 'title'
        """
        passages = []
        
        wikipedia_text = wiki_data.get('original_content', '')
        if wikipedia_text:
            paragraphs = wikipedia_text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) > 50:
                    passages.append({
                        'content': para,
                        'source': 'wikipedia',
                        'metadata': {'type': 'article_paragraph'}
                    })
        
        web_docs = wiki_data.get('web_full_documents', [])
        for doc in web_docs:
            content = doc.get('content', '')
            title = doc.get('title', '')
            if content:
                # Split web document into paragraphs, just like Wikipedia
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        passages.append({
                            'content': f"{title}: {para}",
                            'source': 'web',
                            'metadata': {'type': 'document_paragraph', 'url': doc.get('url', ''), 'title': title}
                        })
        
        self.passages = passages
        self.embeddings = None
        
    def retrieve_top_k(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k most relevant passages using dense retrieval.
        
        Uses embedding-based similarity (cosine) between query and passages.
        Falls back to keyword overlap if embeddings are not cached.
        
        Args:
            query: The query string (typically an RDF triple in format "(subject, predicate, object)")
            k: Number of passages to retrieve (default: 5)
            
        Returns:
            List of top-k passage dicts with keys: 'content', 'source', 'metadata'
        """
        if not self.passages:
            return []
        
        query_embedding = self._get_embedding(query)
        
        scores = []
        for i, passage in enumerate(self.passages):
            if self.embeddings is not None and i < len(self.embeddings):
                score = self._cosine_similarity(query_embedding, self.embeddings[i])
            else:
                score = self._keyword_overlap_score(query, passage['content'])
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:k]]
        
        return [self.passages[i] for i in top_indices]
    
    @network_retry(max_retries=10, max_delay=120)
    def _get_embedding(self, text: str, max_length: int = 500) -> List[float]:
        """
        Get embedding for text using OpenAI embeddings API.
        
        Args:
            text: Text to embed
            max_length: Maximum character length for text (default: 500)
            
        Returns:
            List of embedding values
        """
        if len(text) > max_length:
            text = text[:max_length]
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def _keyword_overlap_score(self, query: str, passage: str) -> float:
        """
        Fallback scoring method using keyword overlap.
        
        Computes Jaccard-like similarity between query and passage terms.
        
        Args:
            query: Query text
            passage: Passage text
            
        Returns:
            Keyword overlap score (0 to 1)
        """
        query_terms = set(query.lower().split())
        passage_terms = set(passage.lower().split())
        if not query_terms:
            return 0.0
        return len(query_terms & passage_terms) / len(query_terms)
    
    def cache_embeddings(self, batch_size: int = 100, max_chars: int = 500) -> None:
        """
        Pre-compute and cache embeddings for all passages.
        
        Improves retrieval efficiency by avoiding per-query embedding generation.
        
        Args:
            batch_size: Number of passages to embed per API call (default: 100)
            max_chars: Maximum character length per passage (default: 500, ~125 tokens)
        """
        if not self.passages:
            return
        
        self.embeddings = []
        for i in range(0, len(self.passages), batch_size):
            batch = []
            for p in self.passages[i:i+batch_size]:
                content = p['content']
                if len(content) > max_chars:
                    content = content[:max_chars]
                batch.append(content)
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                self.embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                logger.error(f"Error caching embeddings batch {i}: {e}")
                self.embeddings.extend([None] * len(batch))

    def save_to_disk(self, cache_path: str) -> None:
        """
        Save the RAG index (passages and embeddings) to disk.
        
        Args:
            cache_path: Directory path where the index will be saved.
        """
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        
        passages_path = os.path.join(cache_path, "passages.pkl")
        embeddings_path = os.path.join(cache_path, "embeddings.npy")
        metadata_path = os.path.join(cache_path, "metadata.json")
        
        with open(passages_path, "wb") as f:
            pickle.dump(self.passages, f)
        
        if self.embeddings is not None:
            np.save(embeddings_path, np.array(self.embeddings))
        
        metadata = {
            "embedding_model": self.embedding_model,
            "num_passages": len(self.passages),
            "num_embeddings": len(self.embeddings) if self.embeddings else 0
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"RAG index saved to {cache_path}: {len(self.passages)} passages, {len(self.embeddings) if self.embeddings else 0} embeddings")

    @classmethod
    def load_from_disk(cls, cache_path: str) -> Optional['GroundTruthRAG']:
        """
        Load a RAG index from disk.
        
        Args:
            cache_path: Directory path where the index is stored.
            
        Returns:
            GroundTruthRAG instance with loaded index, or None if not found.
        """
        passages_path = os.path.join(cache_path, "passages.pkl")
        embeddings_path = os.path.join(cache_path, "embeddings.npy")
        metadata_path = os.path.join(cache_path, "metadata.json")
        
        if not os.path.exists(passages_path) or not os.path.exists(metadata_path):
            logger.debug(f"No cached RAG index found at {cache_path}")
            return None
        
        try:
            with open(passages_path, "rb") as f:
                passages = pickle.load(f)
            
            embeddings = None
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path).tolist()
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            instance = cls(embedding_model=metadata.get("embedding_model", "text-embedding-3-small"))
            instance.passages = passages
            instance.embeddings = embeddings
            
            logger.info(f"RAG index loaded from {cache_path}: {len(passages)} passages, {len(embeddings) if embeddings else 0} embeddings")
            return instance
        except Exception as e:
            logger.error(f"Error loading RAG index from {cache_path}: {e}")
            return None

    @staticmethod
    def get_cache_path_for_entity(entities_cache_dir: str, entity_name: str) -> str:
        """
        Get the cache directory path for a specific entity.
        
        Args:
            entities_cache_dir: Base directory for entity caches.
            entity_name: Name of the entity.
            
        Returns:
            Full path to the entity's cache directory.
        """
        safe_name = entity_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return os.path.join(entities_cache_dir, safe_name)


# ===== corpus =====
def corpus_dir(data_dir, split):
    return os.path.join(data_dir, split)


def entities_file(data_dir, split):
    return os.path.join(data_dir, split, "entities.json")


def entity_titles(entities_file):
    with open(entities_file, encoding="utf-8") as f:
        return [e["title"] for e in json.load(f)]


def load_corpus(data_dir, split, titles):
    """{title: entry} for the given entity titles. Raises if the corpus lacks or has an incomplete entry."""
    folder = corpus_dir(data_dir, split)
    for name, opener in (("reference_corpus.json.xz", lzma.open), ("GT.json", open), ("GT.json.gz", gzip.open)):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with opener(path, "rt", encoding="utf-8") as f:
                corpus = json.load(f)
            break
    else:
        raise FileNotFoundError(f"no reference_corpus.json.xz, GT.json or GT.json.gz in {folder}")
    problems = []
    for t in titles:
        entry = corpus.get(t)
        if entry is None:
            problems.append(f"{t}: not in the corpus")
        elif not entry.get("original_content") or not entry.get("triples") or not entry.get("web_search_results"):
            problems.append(f"{t}: incomplete entry (article, triples or web results missing)")
    if problems:
        raise ValueError(f"reference corpus {folder} does not fit the entity list:\n  " + "\n  ".join(problems[:20]))
    logger.info(f"Reference corpus {folder}: {len(titles)} entities")
    return {t: corpus[t] for t in titles}


# ===== check =====
CAP = 500
# Deduplication: GLM-5.3-Flash on SCADS. Thinking cannot be switched off for this model on SCADS;
# reasoning_effort "none" keeps it short. A call takes ~5 s; SCADS sometimes leaves a long request
# hanging until its ~10 min proxy timeout while a retry goes through at once, hence the short timeout.
DEDUP_GLM_MODEL = "zai-org/GLM-5.3-Flash"
DEDUP_GLM_PARAMS = {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": 32000, "reasoning_effort": "none"}
DEDUP_GLM_RESPONSE_FORMAT = {"type": "json_object"}
DEDUP_GLM_TIMEOUT = 120
DEDUP_GLM_ATTEMPTS = 6
DEDUP_GLM_WORKERS = 4  # SCADS allows few parallel requests per key and model
DEDUP_RPM = 280
csv.field_size_limit(10_000_000)

DEDUP_GLM_PROMPT = """Below is a numbered list of RDF triples (predicate, object) about the entity '{entity}'.
Some triples may repeat a fact already stated by an earlier triple, only in other words. Find them.
Many lists contain no repetitions at all, for example when the triples were written carefully; then the
correct answer is an empty list.

A triple repeats an earlier triple only if both state the same fact: each one implies the other. Wording
does not matter: relation names, synonyms, abbreviations, spelling, formatting, units and language may differ.

The same fact:
- ("country", "Spain") and ("located in country", "Spain")
- ("founded", "2009") and ("founding year", "2009")
- ("country", "USA") and ("country", "United States of America")
- ("height", "100 m") and ("height", "328 ft")
- ("member", "John Smith") and ("member", "John Smith (painter)"): a bracket or apposition that only
  describes the same value adds nothing new

Different facts, even when the value is the same:
- ("birth place", "Tel Aviv") and ("residence", "Tel Aviv"): different relations
- ("winner", X) and ("participant", X): one is weaker
- ("amends", X) and ("amended by", X): opposite directions
- ("located in", "Madrid, Spain") and ("country", "Spain"): one is more specific
- ("instance of", "non-profit association") and ("instance of", "association"): one adds information
- ("release date", "1923") and ("release date", "15 April 1923"): one is more precise
- ("birth place", "Paris") and ("birth place", "Paris, Texas"): the addition names a different thing
- ("diet", "spiders") and ("diet", "insects"): different values
Values that are only related (a city and its country, a person and their spouse) are different facts.

When in doubt, it is not a repetition.

Respond with JSON only:
{{"duplicates": [{{"index": <int>, "duplicate_of": <int>, "reason": "<at most 15 words>"}}]}}
listing ONLY the triples that repeat an earlier one; "duplicate_of" is the index of the earliest triple
stating the same fact. An empty list is a valid answer.

Triples:
{triples}"""
DEDUP_PROMPT = DEDUP_GLM_PROMPT


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def norm(text):
    """Unicode NFKC, every dash as '-', underscores as spaces, collapsed spaces, case-folded."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_submission(path):
    """Rows as dicts with subject, predicate, object; raises ValueError if the file can't be used."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = {c.strip().lower(): c for c in (reader.fieldnames or [])}
            missing = [c for c in ("subject", "predicate", "object") if c not in columns]
            if missing:
                raise ValueError(f"missing column(s) {missing}; found {reader.fieldnames}")
            return [{k: (row.get(columns[k]) or "").strip() for k in ("subject", "predicate", "object")}
                    for row in reader]
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"not a readable UTF-8 CSV file: {e}")


def judge_glm(client, limiter, entity, triples):
    """One call with all of the entity's triples; the model lists only the repetitions.
    Returns {(i, j): True} for every triple j repeating an earlier triple i, the reasons, and the call record."""
    listing = "\n".join(f"{i}. ({t['predicate']}, {t['object']})" for i, t in enumerate(triples))
    prompt = DEDUP_GLM_PROMPT.format(entity=entity, triples=listing)
    last_error = None
    for attempt in range(1, DEDUP_GLM_ATTEMPTS + 1):
        limiter.wait()
        started = time.time()
        try:
            r = client.chat.completions.create(model=DEDUP_GLM_MODEL, messages=[{"role": "user", "content": prompt}],
                                               response_format=DEDUP_GLM_RESPONSE_FORMAT, **DEDUP_GLM_PARAMS)
            if r.choices[0].finish_reason == "length":
                raise ValueError("answer cut off (max_tokens)")
            message = r.choices[0].message
            answer = _parse_json(message.content or "")
            if not isinstance(answer, dict) or not isinstance(answer.get("duplicates"), list):
                raise ValueError(f"no 'duplicates' list in the answer: {(message.content or '')[:200]}")
            verdicts, reasons = {}, {}
            for d in answer["duplicates"]:
                try:
                    j, i = int(d["index"]), int(d["duplicate_of"])
                except (TypeError, ValueError, KeyError):
                    continue
                if 0 <= i < j < len(triples):
                    verdicts[(i, j)] = True
                    reasons.setdefault(f"{i},{j}", str(d.get("reason", "")))
            reasoning = getattr(message, "reasoning_content", None) or (message.model_extra or {}).get("reasoning") or ""
            usage = r.usage.model_dump() if r.usage else {}
            record = {"entity": entity, "triples": len(triples), "same_fact_as": [[i, j] for i, j in sorted(verdicts)],
                      "reasons": reasons, "raw_answer": message.content, "reasoning_chars": len(reasoning),
                      "id": r.id, "model": r.model, "system_fingerprint": r.system_fingerprint,
                      "finish_reason": r.choices[0].finish_reason, "prompt_tokens": usage.get("prompt_tokens"),
                      "completion_tokens": usage.get("completion_tokens"), "seconds": round(time.time() - started, 2),
                      "attempts": attempt, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            return verdicts, reasons, [record]
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(5 * attempt)
    raise RuntimeError(last_error)


# ===== score =====
JUDGE = "google/gemma-4-26B-A4B-it"
SEED = 42
SAMPLE_SIZE = 1000
TOP_K = 10
JUDGE_WORKERS = 8
JUDGE_RPM = 280  # the SCADS key allows 300 requests per minute
PRECISION = "Precision (Precision (RAG))"  # metric labels as in the BeQu result files
RECALL = "Recall (Wikipedia+Web)"
SOURCE = "elicited_triples"  # the submission's name in the detailed results (BeQu: the CSV file name)
DETAILED_FIELDS = ["subject", "predicate", "object", "result", "result_category", "reasoning", "metric",
                   "source_file", "retrieved_passages"]
RESULT_FIELDS = ["Entailment", "Contradiction", "Neutral", "Entailment_ratio", "Contradiction_ratio",
                 "Neutral_ratio", "Total #Triples", "Error_count", "Error_ratio", "Metric", "Source Elicited File",
                 "Source Prompt File"]
CATEGORY_NAMES = {"a": "Entailment", "b": "Contradiction", "c": "Neutral", "error": "Error"}


class Scorer:
    def __init__(self, triples, corpus, rag_cache_dir, results_dir, judge=JUDGE, judge_api_url=None,
                 judge_api_key=None, seed=SEED, sample_size=SAMPLE_SIZE, top_k=TOP_K, judge_workers=JUDGE_WORKERS):
        self.triples = triples          # checked submission rows: subject, predicate, object, subject_name
        self.corpus = corpus            # {title: reference entry}
        self.rag_cache_dir = rag_cache_dir
        self.results_dir = results_dir
        self.judge, self.judge_api_url, self.judge_api_key = judge, judge_api_url, judge_api_key
        self.sample_size, self.top_k, self.judge_workers = sample_size, top_k, max(1, judge_workers)
        # one generator for both samples, precision first: changing the order changes both samples
        self._rng = random.Random(int(seed))
        self.aggregated = []
        self._resume_lock, self._calls_lock = threading.Lock(), threading.Lock()
        self._calls_path = os.path.join(results_dir, "judge_calls.jsonl")

    # --- judge client, logging every call ---------------------------------------------------------
    def _request(self):
        request = Request(self.judge, judge_api_url=self.judge_api_url, judge_api_key=self.judge_api_key)
        request.call_log = self._log_call
        return request

    def _log_call(self, kind, triple, response, seconds):
        u = response.usage
        row = {"metric": kind, "triple": triple, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "seconds": round(seconds, 2), "id": response.id, "model": response.model,
               "system_fingerprint": response.system_fingerprint,
               "prompt_tokens": u.prompt_tokens if u else None, "completion_tokens": u.completion_tokens if u else None,
               "finish_reason": response.choices[0].finish_reason}
        with self._calls_lock, open(self._calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- sampling (stratified: a random entity, then a random triple of it) -------------------------
    def sample_grouped(self, subj_to_triples, m):
        available = {s: list(v) for s, v in subj_to_triples.items() if v}
        subjects = list(available.keys())
        sampled = []
        if not subjects:
            return sampled
        attempts = 0
        while len(sampled) < m and subjects:
            subj = self._rng.choice(subjects)
            trio_list = available.get(subj)
            if not trio_list:
                subjects.remove(subj)
                continue
            triple = self._rng.choice(trio_list)
            if triple not in sampled:
                sampled.append(triple)
                trio_list.remove(triple)
                if not trio_list:
                    subjects.remove(subj)
            else:  # an identical triple twice: drop it from the pool
                try:
                    trio_list.remove(triple)
                except ValueError:
                    pass
                if not trio_list:
                    subjects.remove(subj)
            attempts += 1
            if attempts > m * 50:
                break
        return sampled

    def sample(self, triples, m):
        grouped = {}
        for t in triples:
            grouped.setdefault(t.get("subject"), []).append(t)
        return self.sample_grouped(grouped, m)

    # --- resuming: judgments of an interrupted run are reused --------------------------------------
    def _open_detailed(self):
        path = os.path.join(self.results_dir, "results_detailed.csv")
        finished = os.path.exists(os.path.join(self.results_dir, "results.csv"))
        if os.path.exists(path) and not finished:
            os.rename(path, os.path.join(self.results_dir,
                                         f"results_detailed.interrupted_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
        self._cache, self._reused, self._interrupted = {}, 0, []
        if not finished:
            csv.field_size_limit(10**9)
            self._interrupted = sorted(glob.glob(os.path.join(self.results_dir, "results_detailed.interrupted_*.csv")))
            for p in self._interrupted:  # oldest first; a newer file overrides what it contains
                rows = defaultdict(list)
                try:
                    with open(p, newline="", encoding="utf-8") as fh:
                        for row in csv.DictReader(fh, escapechar="\\"):
                            if row.get("result") in ("a", "b", "c") and row.get("metric"):
                                rows[(row["metric"], row.get("subject", ""), row.get("predicate", ""),
                                      row.get("object", ""))].append(row)
                except Exception as e:  # a half-written last line after a kill
                    logger.warning(f"Stopped reading {p} early ({e}); using the rows read so far")
                self._cache.update(rows)
            if self._cache:
                logger.info(f"Resuming: {sum(len(v) for v in self._cache.values())} judgments will be reused")
        self._detailed_file = open(path, "w", newline="", encoding="utf-8")
        self._detailed = csv.DictWriter(self._detailed_file, fieldnames=DETAILED_FIELDS,
                                        quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        self._detailed.writeheader()
        self._detailed_file.flush()

    def _cached(self, metric, subject, predicate, obj):
        with self._resume_lock:
            rows = self._cache.get((metric, subject, predicate, obj))
            if not rows:
                return None
            self._reused += 1
            return rows.pop(0)

    def _write_detailed(self, row):
        result = row.get("result", "c")
        self._detailed.writerow({**{k: row.get(k, "") for k in DETAILED_FIELDS}, "result": result,
                                 "result_category": CATEGORY_NAMES.get(result, "Unknown")})
        self._detailed_file.flush()

    def _map_in_order(self, fn, items):
        if self.judge_workers <= 1:
            yield from map(fn, items)
            return
        with ThreadPoolExecutor(max_workers=self.judge_workers) as executor:
            yield from executor.map(fn, items)

    # --- precision ----------------------------------------------------------------------------------
    def _rag_index(self, subject_name, entry):
        """The entity's passage index, from rag_cache or built (embeddings via OpenRouter) and cached."""
        cache_path = GroundTruthRAG.get_cache_path_for_entity(self.rag_cache_dir, subject_name)
        rag = GroundTruthRAG.load_from_disk(cache_path)
        if rag is not None:  # the normal case: the index ships with the corpus, nothing is written
            return rag
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        fd = os.open(cache_path + ".lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                rag = GroundTruthRAG.load_from_disk(cache_path)
                if rag is None:
                    rag = GroundTruthRAG()
                    rag.build_index(entry)
                    rag.cache_embeddings()
                    try:
                        rag.save_to_disk(cache_path)
                    except Exception as e:
                        logger.warning(f"Failed to save RAG cache for {subject_name}: {e}")
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return rag

    def _precision_for_entity(self, request, subject_name, entity_triples, entry):
        """[(category or None for an error, detailed row, triple)] in the order of entity_triples."""
        def row(t, result, reasoning, passages=""):
            return {"subject": t.get("subject", ""), "predicate": t.get("predicate", ""), "object": t.get("object", ""),
                    "result": result, "reasoning": reasoning, "metric": PRECISION, "source_file": SOURCE,
                    "retrieved_passages": passages}
        if not entry:
            return [("c", row(t, "c", "No Wikipedia/Web data found for subject"), t) for t in entity_triples]
        try:
            rag = self._rag_index(subject_name, entry)
        except Exception as e:
            logger.error(f"Error building RAG index for {subject_name}: {e}")
            return [(None, row(t, "error", f"RAG index build error: {e}"), t) for t in entity_triples]
        rows = []
        for t in entity_triples:
            subject, predicate, obj = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
            triple_str = f"({subject}, {predicate}, {obj})"
            cached = self._cached(PRECISION, subject, predicate, obj)
            if cached is not None:
                rows.append((cached["result"], cached, t))
                continue
            passages_str = ""
            try:
                passages = rag.retrieve_top_k(triple_str, k=self.top_k)
                if not passages:
                    category, reasoning = "c", "No relevant passages found in ground truth"
                else:
                    passages_text = "\n\n".join(f"[{p['source'].upper()}] {p['content'][:500]}" for p in passages)
                    output = request.verify_triple_with_rag(triple_str, passages_text, [p["source"] for p in passages])
                    category = output.get("answer", "c") if isinstance(output, dict) else "c"
                    reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                    # sorted: the research pipeline joins a set here, whose order changes from run to run
                    reasoning = f"RAG retrieved: {', '.join(sorted(set(p['source'] for p in passages)))} | {reasoning}"
                passages_str = "\n\n---\n\n".join(f"[{p['source'].upper()}] {p['content'][:2000]}"
                                                  for p in passages) if passages else ""
                rows.append((category, row(t, category, reasoning, passages_str), t))
            except Exception as e:
                logger.error(f"Error verifying triple {triple_str}: {e}")
                rows.append((None, row(t, "error", f"Verification error after retries: {e}", passages_str), t))
        return rows

    def precision(self):
        request = self._request()
        sampled = self.sample(self.triples, self.sample_size)
        by_entity = defaultdict(list)
        for t in sampled:
            if t.get("subject_name", ""):
                by_entity[t["subject_name"]].append(t)
        results = {"a": 0, "b": 0, "c": 0}
        judge = lambda item: self._precision_for_entity(request, item[0], item[1], self.corpus.get(item[0]))
        for entity_rows in tqdm(self._map_in_order(judge, by_entity.items()), total=len(by_entity), desc="Precision"):
            for category, row, _ in entity_rows:
                if category is not None:
                    results[category] += 1
                self._write_detailed(row)
        self._aggregate(results, len(sampled), PRECISION)

    # --- recall -------------------------------------------------------------------------------------
    def _recall_for_fact(self, request, fact, by_subject):
        subject, predicate, obj = fact.get("subject", ""), fact.get("predicate", ""), fact.get("object", "")
        row = {"subject": subject, "predicate": predicate, "object": obj, "metric": RECALL, "source_file": SOURCE,
               "retrieved_passages": ""}
        elicited = by_subject.get(subject, [])
        if not elicited:
            return "c", {**row, "result": "c", "reasoning": "No elicited triples found for subject"}
        cached = self._cached(RECALL, subject, predicate, obj)
        if cached is not None:
            return cached["result"], cached
        fact_str = f"({subject}, {predicate}, {obj})"
        try:
            output = request.verify_triple_lm_wikidata(fact_str, ", ".join(elicited))
            category = output.get("answer", "c") if isinstance(output, dict) else "c"
            return category, {**row, "result": category,
                              "reasoning": output.get("reasoning", "") if isinstance(output, dict) else ""}
        except Exception as e:
            logger.error(f"Error verifying triple {fact_str}: {e}")
            return None, {**row, "result": "error", "reasoning": f"Verification error after retries: {e}"}

    def recall(self):
        request = self._request()
        facts = {}
        for title, entry in sorted(self.corpus.items()):  # sorted: the seeded sample must not depend on load order
            all_facts = list(entry.get("triples", []) or []) + list(entry.get("web_triples", []) or [])
            if all_facts:
                facts[title] = all_facts
        sampled = self.sample_grouped(facts, self.sample_size)
        by_subject = defaultdict(list)  # the submission's triples per subject, as the judge sees them
        for t in self.triples:
            by_subject[t["subject"]].append(f"({t['subject']}, {t['predicate']}, {t['object']})")
        results = {"a": 0, "b": 0, "c": 0}
        judge = lambda fact: self._recall_for_fact(request, fact, by_subject)
        for category, row in tqdm(self._map_in_order(judge, sampled), total=len(sampled), desc="Recall"):
            if category is not None:
                results[category] += 1
            self._write_detailed(row)
        self._aggregate(results, len(sampled), RECALL)

    def _aggregate(self, results, total, metric):
        if not total:
            return
        errors = total - sum(results.values())
        self.aggregated.append({"Entailment": results["a"], "Contradiction": results["b"], "Neutral": results["c"],
                                "Entailment_ratio": results["a"] / total, "Contradiction_ratio": results["b"] / total,
                                "Neutral_ratio": results["c"] / total, "Total #Triples": total,
                                "Error_count": errors, "Error_ratio": errors / total, "Metric": metric,
                                "Source Elicited File": SOURCE})
        logger.info(f"{metric}: entailed {results['a']}, contradicted {results['b']}, neutral {results['c']}, "
                    f"errors {errors} of {total}")

    def run(self):
        os.makedirs(self.results_dir, exist_ok=True)
        self._open_detailed()
        try:
            self.precision()
            self.recall()
        finally:
            self._detailed_file.close()
        with open(os.path.join(self.results_dir, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            w.writeheader()
            w.writerows(self.aggregated)
        if self._reused:
            logger.info(f"Reused {self._reused} judgments of an interrupted run")
        for p in self._interrupted:
            os.remove(p)
        p = next((r for r in self.aggregated if r["Metric"] == PRECISION), None)
        r = next((r for r in self.aggregated if r["Metric"] == RECALL), None)
        precision = p["Entailment_ratio"] if p else 0.0
        recall = r["Entailment_ratio"] if r else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1,
                "complete": bool(p and r) and p["Error_count"] == 0 and r["Error_count"] == 0,
                "judgments": {"precision": p, "recall": r}}


def read_checked(checked_dir):
    path = os.path.join(checked_dir, "elicited_triples.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_submission(submission, split, out, data_dir):
    """Steps 1-6 of the checking. Writes <out>/elicited_triples.csv, validation_report.json, dedup_calls.jsonl."""
    entities_file_path = entities_file(data_dir, split)
    entities = [e["title"] for e in json.load(open(entities_file_path, encoding="utf-8"))]
    by_norm = {norm(t): t for t in entities}
    rows = read_submission(submission)  # ValueError: the submission is rejected

    # steps 2-4
    dropped, counts = [], Counter(rows_read=len(rows))
    per_entity, seen = defaultdict(list), defaultdict(set)
    for n, row in enumerate(rows, start=2):  # line numbers in the file (header is line 1)
        entity = by_norm.get(norm(row["subject"]))
        if not all(row.values()):
            reason = "empty field"
        elif entity is None:
            reason = "subject is not an entity of the split"
        elif (key := (norm(row["predicate"]), norm(row["object"]))) in seen[entity]:
            reason = "exact duplicate"
        elif len(per_entity[entity]) >= CAP:
            reason = f"over the cap of {CAP} triples per entity"
        else:
            if row["subject"] != entity:
                counts["subject matched after normalising"] += 1
            seen[entity].add(key)
            per_entity[entity].append({**row, "subject": entity})
            continue
        counts[reason] += 1
        dropped.append({"line": n, "reason": reason, **row})

    # step 5, resumable from dedup_calls.jsonl
    os.makedirs(out, exist_ok=True)
    calls_path = os.path.join(out, "dedup_calls.jsonl")
    fingerprint = lambda e: hashlib.sha256(json.dumps(per_entity[e], ensure_ascii=False).encode()).hexdigest()
    done, done_reasons = {}, {}  # (entity, input sha) -> {(i, j): True} / {"i,j": reason}
    if os.path.exists(calls_path):
        for line in open(calls_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # line cut off when a run was stopped
                continue
            if r.get("entity_done"):
                done[(r["entity"], r["input_sha256"])] = {(i, j): v for i, j, v in r["verdicts"]}
                done_reasons[(r["entity"], r["input_sha256"])] = r.get("reasons", {})
    todo = [e for e in entities if len(per_entity[e]) > 1 and (e, fingerprint(e)) not in done]
    print(f"{len(rows)} rows read; deduplication: {len(todo)} entities to call", flush=True)
    client = openai.OpenAI(base_url=api_base_url(), api_key=api_key(),
                           timeout=DEDUP_GLM_TIMEOUT, max_retries=0)
    limiter = RateLimiter(DEDUP_RPM)

    def run(entity):
        try:
            verdicts, reasons, records = judge_glm(client, limiter, entity, per_entity[entity])
            return records + [{"entity": entity, "entity_done": True, "input_sha256": fingerprint(entity),
                               "verdicts": [[i, j, v] for (i, j), v in sorted(verdicts.items())],
                               "reasons": reasons}]
        except RuntimeError as e:
            return [{"entity": entity, "error": str(e), "input_sha256": fingerprint(entity),
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}]

    with open(calls_path, "a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=DEDUP_GLM_WORKERS) as pool:
        for k, lines in enumerate(pool.map(run, todo), 1):
            for record in lines:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                if record.get("entity_done"):
                    key = (record["entity"], record["input_sha256"])
                    done[key] = {(i, j): v for i, j, v in record["verdicts"]}
                    done_reasons[key] = record["reasons"]
            log.flush()
            if k % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} deduplication {k}/{len(todo)}", flush=True)

    # step 6
    reason_same = "same fact as an earlier triple"
    out_rows, failed, entity_rows = [], [], []
    for e in entities:
        triples = per_entity[e]
        key = (e, fingerprint(e))
        verdicts = done.get(key) if len(triples) > 1 else {}
        if verdicts is None:
            failed.append(e)
            verdicts = {}
        same_as = {}
        for (i, j), same in sorted(verdicts.items()):
            if same and j not in same_as:
                same_as[j] = i
        reasons = done_reasons.get(key, {})
        for i, t in enumerate(triples):
            if i in same_as:
                counts[reason_same] += 1
                dropped.append({"reason": reason_same, **t, "same_fact_as": triples[same_as[i]],
                                "model_reason": reasons.get(f"{same_as[i]},{i}", "")})
            else:
                out_rows.append({**t, "subject_name": e})
        entity_rows.append({"entity": e, "after_cap": len(triples), "removed_as_same_fact": len(same_as),
                            "kept": len(triples) - len(same_as)})
    counts["kept"] = len(out_rows)

    out_csv = os.path.join(out, "elicited_triples.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "subject_name"])
        w.writeheader()
        w.writerows(out_rows)

    report = {
        "submission": os.path.abspath(submission), "submission_sha256": sha256(submission),
        "split": split, "entities_file_sha256": sha256(entities_file_path),
        "checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": {"cap_per_entity": CAP, "order": "cap first, then same-fact deduplication",
                     "subject_matching": "exact title after NFKC, dash, space and case normalising",
                     "exact_duplicates": "predicate and object equal after the same normalising, underscores as spaces",
                     "dedup_model": DEDUP_GLM_MODEL, "dedup_api": "SCADS", "dedup_parameters": DEDUP_GLM_PARAMS,
                     "dedup_response_format": DEDUP_GLM_RESPONSE_FORMAT,
                     "dedup_request_timeout_s": DEDUP_GLM_TIMEOUT, "dedup_attempts": DEDUP_GLM_ATTEMPTS,
                     "dedup_prompt": DEDUP_PROMPT,
                     "dedup_prompt_sha256": hashlib.sha256(DEDUP_PROMPT.encode()).hexdigest()},
        "evaluate_py_sha256": sha256(os.path.abspath(__file__)),
        "complete": not failed, "dedup_failed_entities": failed,
        "counts": dict(counts),
        "entities_with_triples": sum(r["kept"] > 0 for r in entity_rows),
        "entities_without_triples": [r["entity"] for r in entity_rows if r["kept"] == 0],
        "output_csv_sha256": sha256(out_csv),
        "per_entity": entity_rows,
        "dropped": dropped,
    }
    with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("complete", "counts", "entities_with_triples")}
                     | {"dedup_failed_entities": len(failed)}, indent=2, ensure_ascii=False))
    return report


def score_submission(out, split, data_dir):
    """Precision, recall and F1 of the checked submission in <out>; results in <out>/scores/."""
    results_dir = os.path.join(out, "scores")
    os.makedirs(results_dir, exist_ok=True)
    logger.add(os.path.join(results_dir, "scoring.log"), level="DEBUG")
    set_judge_rate_limit(JUDGE_RPM)
    corpus = load_corpus(data_dir, split, entity_titles(entities_file(data_dir, split)))
    triples = read_checked(out)
    started = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    scores = Scorer(triples, corpus, os.path.join(data_dir, split, "rag_cache"), results_dir).run()
    scores.update({"split": split, "started": started, "finished": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                   "settings": {"judge": JUDGE, "judge_api": api_base_url(), "seed": SEED, "sample_size": SAMPLE_SIZE,
                                "top_k": TOP_K, "embedding_model": "openai/text-embedding-3-small (OpenRouter)",
                                "judge_workers": JUDGE_WORKERS, "judge_rpm": JUDGE_RPM},
                   "evaluate_py_sha256": sha256(os.path.abspath(__file__))})
    with open(os.path.join(results_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    return scores


def main():
    ap = argparse.ArgumentParser(description="Evaluate an AKBC 2027 shared task submission: check it, then score it.")
    ap.add_argument("submission", help="CSV with columns subject, predicate, object")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", required=True, help="output folder (rerun with the same folder to resume)")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"),
                    help="folder with <split>/entities.json and <split>/reference_corpus.json.xz (default: ./data)")
    ap.add_argument("--check_only", action="store_true", help="run the checking only, no scoring")
    args = ap.parse_args()
    if not api_base_url() or not api_key():
        raise SystemExit("set API_BASE_URL and API_KEY (or SCADSAI_BASE_URL and SCADSAI_API_KEY)")

    try:
        report = check_submission(args.submission, args.split, args.out, args.data_dir)
    except ValueError as e:
        print(f"REJECTED: {e}")
        sys.exit(2)
    if not report["complete"]:
        print(f"INCOMPLETE: deduplication failed for {len(report['dedup_failed_entities'])} entities; "
              f"run the same command again")
        sys.exit(1)
    if args.check_only:
        return
    scores = score_submission(args.out, args.split, args.data_dir)
    print(json.dumps({k: scores[k] for k in ("precision", "recall", "f1", "complete")}, indent=2))
    if not scores["complete"]:
        print("INCOMPLETE: some judgments failed; run the same command again")
        sys.exit(1)


if __name__ == "__main__":
    main()
