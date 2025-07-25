
# -*- coding: utf-8 -*-

import re
import unicodedata
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
from langdetect import detect, LangDetectException
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class TextChunk:
    content: str
    chunk_id: str
    source_page: int
    chunk_index: int
    language: str
    content_type: str
    word_count: int
    char_count: int


class TextPreprocessor:
    def __init__(self, chunk_size=512, min_chunk_size=100):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.bangla_sent = re.compile(r'[।!?]+\s*')
        self.en_sent = re.compile(r'[.!?]+\s+')
        self.mcq_pattern = re.compile(
            r'([কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়A-Da-d৪১-৪][\)\. ]\s*)', re.MULTILINE)
        self.paragraph_sep = re.compile(r'\n\s*\n+')

    def detect_language(self, text: str) -> str:
        if not text or len(text.strip()) < 10:
            return 'unknown'
        bangla = len(re.findall(r'[\u0980-\u09FF]', text))
        total = len(re.findall(r'[A-Za-z\u0980-\u09FF]', text))
        ratio = bangla / total if total > 0 else 0
        if ratio > 0.7:
            return 'bn'
        elif ratio > 0.3:
            return 'mixed'
        try:
            d = detect(text)
            return d if d in ('bn', 'en') else 'en'
        except LangDetectException:
            return 'en'

    def identify_content_type(self, text: str) -> str:
        t = text.strip().lower()
        lines = t.split('\n')
        if any(ind in t for ind in ['উত্তর', 'answer']):
            return 'answer_table'
        if self.mcq_pattern.search(text):
            return 'mcq'
        if '|' in text and text.count('|') > 2 or (len(lines) > 2 and all(len(l.split()) > 2 for l in lines[:3])):
            return 'table'
        if len(text) < 150 and re.search(r'^\d+\.\d+|^অধ্যায়|^পরিচ্ছেদ', text):
            return 'heading'
        if re.search(r'^[\s]*[\-\*•]\s', text, re.MULTILINE):
            return 'list'
        if any(sym in text for sym in '=+×÷∑∏') and len(text) < 200:
            return 'equation'
        if any(w in t for w in ['সংজ্ঞা', 'meaning', 'definition']) and len(text) < 300:
            return 'definition'
        return 'paragraph'

    def clean_text(self, text: str) -> str:
        t = unicodedata.normalize("NFC", text)
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'\n\s*\n', '\n\n', t)
        t = t.replace('।।', '।').replace('??', '?').replace('!!', '!')
        t = re.sub(r'\s+([।!?,.;:])', r'\1', t)
        t = re.sub(r'([।!?])\s*', r'\1 ', t)
        t = re.sub(r'\s*\(\s*', ' (', t)
        t = re.sub(r'\s*\)\s*', ') ', t)
        return t.strip()

    def split_sentences(self, text: str, lang: str) -> List[str]:
        sents = []
        if lang in ('bn', 'mixed'):
            sents += [s.strip() for s in self.bangla_sent.split(text) if s.strip()]
        if lang in ('en', 'mixed'):
            try:
                sents += nltk.sent_tokenize(text)
            except Exception:
                sents += [s.strip() for s in self.en_sent.split(text) if s.strip()]
        if not sents:
            sents = [s.strip() for s in re.split(r'[।.!?]+', text) if s.strip()]
        return sents

    def chunk_by_sentences(self, text: str, page: int) -> List[TextChunk]:
        cleaned = self.clean_text(text)
        lang = self.detect_language(cleaned)
        ctype = self.identify_content_type(cleaned)
        sents = self.split_sentences(cleaned, lang)
        chunks, curr, idx = [], "", 0
        for sent in sents:
            cand = f"{curr} {sent}".strip() if curr else sent
            if len(cand) > self.chunk_size and curr:
                chunks.append(self._make_chunk(curr, page, idx, lang, ctype))
                idx += 1
                curr = sent
            else:
                curr = cand
        if curr:
            chunks.append(self._make_chunk(curr, page, idx, lang, ctype))
        return chunks

    def _make_chunk(self, content, page, idx, lang, ctype) -> TextChunk:
        return TextChunk(
            content=content.strip(),
            chunk_id=f"page_{page}_{idx}",
            source_page=page,
            chunk_index=idx,
            language=lang,
            content_type=ctype,
            word_count=len(content.split()),
            char_count=len(content)
        )

    def process_document(self, doc: Dict[str, Any]) -> List[TextChunk]:
        chunks, seen = [], set()
        for pg in doc.get('pages', []):
            txt = pg.get('text', '')
            if not txt:
                continue
            for chunk in self.chunk_by_sentences(txt, pg.get('page_number', 0)):
                h = hash(chunk.content.strip().lower())
                if chunk.char_count >= self.min_chunk_size and h not in seen:
                    seen.add(h)
                    chunks.append(chunk)
        logger.info(f"Processed into {len(chunks)} chunks")
        return chunks


def preprocess_document(document_data: Dict[str, Any], chunk_size: int = 512) -> List[TextChunk]:
    pp = TextPreprocessor(chunk_size=chunk_size)
    return pp.process_document(document_data)
