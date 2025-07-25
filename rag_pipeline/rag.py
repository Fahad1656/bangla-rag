
# -*- coding: utf-8 -*-
import os

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langdetect import detect, LangDetectException
from loguru import logger


RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))

@dataclass
class QAResult:
    answer: str
    sources: List[Dict[str, Any]]
    language: str
    model: str
    confidence: float

class LanguageID:
    def __init__(self):
        self.bn_keys = ['কি', 'কী', 'কে', 'কোথায়', 'কেন', 'কিভাবে', 'কখন', 'বাংলা', 'বাংলাদেশ', 'ঢাকা', 'এর', 'এবং', 'তার', 'যে']
        self.en_keys = ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'the', 'and']

    def detect(self, text: str) -> str:
        if not text or len(text.strip()) < 3: return 'en'
        text_lower = text.lower()
        bn_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
        if total_chars == 0: return 'en'

        bn_ratio = bn_chars / total_chars
        bn_hits = sum(1 for k in self.bn_keys if k in text_lower)
        en_hits = sum(1 for k in self.en_keys if k in text_lower)

        if bn_ratio > 0.5 or bn_hits > en_hits: return 'bn'
        elif bn_ratio > 0.2: return 'mixed'
        try: return 'bn' if detect(text) == 'bn' else 'en'
        except LangDetectException: return 'en'

class OllamaLLM:
    def __init__(self, model=RAG_MODEL_NAME, base=OLLAMA_API_URL, temp=0.1):
        self.model = model
        self.base = base
        self.temp = temp
        try:
            self.client = Ollama(model=model, base_url=base, temperature=temp)
        except Exception as e:
            logger.error(f"Ollama init failed: {e}")
            raise

    def reply(self, prompt: str) -> str:
        return self.client.invoke(prompt).strip()

    def test(self) -> bool:
        try:
            return bool(self.reply("Hello"))
        except:
            return False

class SimplePrompt:
    def __init__(self):
        self.template = PromptTemplate(
            template="""
প্রসঙ্গ: {context}

🚨 CRITICAL INSTRUCTIONS:
- Answer ONLY the question asked below
- Give ONLY 1-2 words as answer
- Do NOT repeat the examples
- Do NOT explain anything

Examples:
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? → A: শুম্ভুনাথ
Q: কাকে অনুপমের ভাগ্য দেবতা বলা হয়েছে? → A: মামাকে  
Q: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? → A: ১৫ বছর

প্রশ্ন: {question}
উত্তর: """,
            input_variables=["context", "question"]
        )

    def format(self, ctx, qn):
        return self.template.format(context=ctx, question=qn)

class MiniRAG:
    def __init__(self, vector_store, embedder, model=RAG_MODEL_NAME, base_url=OLLAMA_API_URL):
        self.vs = vector_store
        self.emb = embedder
        self.lang = LanguageID()
        self.llm = OllamaLLM(model, base_url)
        self.prompt = SimplePrompt()
        self.memory = ConversationBufferWindowMemory(k=4, return_messages=True)

    def _search_context(self, q: str, k=5, thresh=0.3) -> Tuple[str, List[Dict]]:
        try:
            results = self.vs.search_by_text(q, self.emb, k=k, threshold=thresh)
            context = "\n\n".join(r.content for r in results)
            sources = [{"chunk_id": r.chunk_id, "content": r.content, "score": r.score, "meta": r.metadata} for r in results]
            return context, sources
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return "", []

    def _clean_answer(self, text: str) -> str:
        if "উত্তর:" in text:
            text = text.split("উত্তর:")[-1].strip()
        if 'প্রশ্ন:' in text:
            lines = [l for l in text.split('\n') if l.strip().startswith("উত্তর:")]
            if lines: text = lines[-1].split("উত্তর:")[1].strip()
        return ' '.join(text.strip().split()[:3])

    def _confidence(self, chunks: List[Dict]) -> float:
        if not chunks: return 0.0
        score = sum(c['score'] for c in chunks) / len(chunks)
        length_factor = min(sum(len(c['content']) for c in chunks) / 1000.0, 1.0)
        return round(min((score * 0.6 + 0.2 * len(chunks) / 3 + 0.2 * length_factor), 1.0), 3)

    def get_answer(self, qn: str, top_k=5, threshold=0.3) -> QAResult:
        logger.info(f"Q: {qn}")
        try:
            lang = self.lang.detect(qn)
            ctx, sources = self._search_context(qn, k=top_k, thresh=threshold)
            if not ctx:
                return QAResult("No matching content found.", [], lang, "none", 0.0)

            prompt = self.prompt.format(ctx, qn)
            response = self.llm.reply(prompt)
            answer = self._clean_answer(response)

            self.memory.chat_memory.add_user_message(qn)
            self.memory.chat_memory.add_ai_message(answer)

            return QAResult(answer, sources, lang, self.llm.model, self._confidence(sources))

        except Exception as e:
            logger.error(f"RAG failed: {e}")
            return QAResult("Error during answer generation.", [], 'en', "error", 0.0)

    def clear_chat(self):
        self.memory.clear()

    def history(self) -> List[Dict[str, str]]:
        msgs = self.memory.chat_memory.messages
        return [{"question": msgs[i].content, "answer": msgs[i+1].content} for i in range(0, len(msgs) - 1, 2)]

    def test_all(self) -> Dict[str, Any]:
        return {
            "ollama_ok": self.llm.test(),
            "vector_store": self.vs.get_statistics(),
            "embedder_ok": bool(self.emb.encode_query("test")),
            "memory_ok": True
        }
