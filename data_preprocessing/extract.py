# -*- coding: utf-8 -*-
import unicodedata
import re
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any

from bangla_pdf_ocr import process_pdf  # unified PDF to pages extractor





def extract_pdf_content(pdf_path: str, output_dir: str = None) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Fix: ensure structured result
    result = process_pdf(str(pdf_path))

    if isinstance(result, str):
        # process_pdf returned plain text, wrap it manually
        logger.warning("process_pdf() returned string, converting to pages manually.")
        result = {
            "pages": [{"page_number": 1, "text": result}],
            "source": str(pdf_path)
        }

    full_text = "\n\n".join(p.get("text", "") for p in result.get("pages", []))
    content_analysis = _identify_content_types(full_text)
    stats = {
        "total_pages": len(result.get("pages", [])),
        "total_chars": len(full_text),
        "total_words": len(full_text.split()),
    }

    return {
        "pages": result.get("pages", []),
        "full_text": full_text,
        "content_analysis": content_analysis,
        "statistics": stats,
        "source_file": str(pdf_path)
    }

def _identify_content_types(text: str) -> Dict[str, bool]:
    ct = {'has_mcq': False, 'has_bangla': False, 'has_english': False,
          'has_numbers': False, 'has_equations': False}
    if re.search(r'[ক-হ]\)|[a-d]\)|\([ক-হ]\)|\([a-d]\)|[A-D]\.|[১-৪]\.', text):
        ct['has_mcq'] = True
    if re.search(r'[\u0980-\u09FF]', text):
        ct['has_bangla'] = True
    if re.search(r'[A-Za-z]', text):
        ct['has_english'] = True
    if re.search(r'[০-৯0-9]|[+\-×÷=<>∑∏]', text):
        ct['has_numbers'] = True
    if re.search(r'\$.*?\$|\\[a-zA-Z]+|[=≈≤≥]', text):
        ct['has_equations'] = True
    return ct
