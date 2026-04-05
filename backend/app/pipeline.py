import os
import shutil
import tempfile
import re
import asyncio
import logging
import requests
from bs4 import BeautifulSoup
from typing import Optional
from fastapi import UploadFile
from app.axes.authenticity import Authenticity

logger = logging.getLogger("Pipeline")

class Pipeline:
    def __init__(self):
        self.auth = Authenticity()
        # Initialize temp directory
        self.temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)


    async def run(self, input_type: str, file: Optional[UploadFile] = None, url: Optional[str] = None) -> dict:
        """
        Coordinates preprocessing and axis evaluation for various input types.
        """
        features = {"type": input_type}
        temp_path = None

        try:
            # 1. ── Save File if provided ──
            if file:
                filename = f"upload_{os.urandom(4).hex()}_{file.filename}"
                temp_path = os.path.join(self.temp_dir, filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                features["path"] = temp_path

            # 2. ── Preprocessing based on type ──
            if input_type == "image":
                # Path/URL is already handled
                if url: features["url"] = url

            elif input_type == "video":
                # Basic metadata extraction
                if temp_path:
                    # frames_count, fps, face_detected, etc.
                    # For now we'll let Authenticity.analyze_video_frequency_hybrid handle path directly.
                    features["video_path"] = temp_path
                elif url:
                    features["video_url"] = url

            elif input_type == "document":
                # Extract text for RoBERTa & Burstiness
                if temp_path:
                    text_content = self._extract_document_text(temp_path)
                    features["clean_text"] = text_content
                    features["word_count"] = len(text_content.split())
                    
                    # Calculate writing metrics for Signal 2 (Burstiness)
                    metrics = self._calculate_writing_metrics(text_content)
                    features.update(metrics)
                    
                    # Optional: Basic metadata collection (simulated)
                    features["metadata"] = {
                        "filename": file.filename if file else "unknown",
                        "size": os.path.getsize(temp_path) if temp_path else 0
                    }
                
            elif input_type == "url" and url:
                # Scrape article body
                features["url"] = url
                body_text = self._scrape_url(url)
                features["body_text"] = body_text
                features["word_count"] = len(body_text.split())
                
                # Calculate metrics for the scraped text
                metrics = self._calculate_writing_metrics(body_text)
                features.update(metrics)

            # 3. ── Evaluate Authenticity ──
            result = self.auth.evaluate(features)

            return {
                "authenticity": result,
                # "consistency": self.cons.evaluate(features),
                # "credibility": self.cred.evaluate(features),
            }

        finally:
            # Cleanup temp file? 
            # pass (Keep for now to debug)
            pass

    def _calculate_writing_metrics(self, text: str) -> dict:
        """
        Calculates linguistic metrics for burstiness analysis.
        """
        import numpy as np
        # Split by common sentence delimiters
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        if not sentences:
            return {
                "burstiness_ratio": 0.0,
                "sentence_length_variance": 0.0,
                "avg_sentence_length": 0.0
            }

        # Calculate word counts per sentence
        counts = [len(s.split()) for s in sentences]
        
        avg_len = np.mean(counts)
        std_len = np.std(counts)
        variance = np.var(counts)
        
        # Burstiness Ratio = StdDev / Mean (Uniformity coefficient)
        ratio = std_len / avg_len if avg_len > 0 else 0
        
        return {
            "burstiness_ratio": round(float(ratio), 4),
            "sentence_length_variance": round(float(variance), 4),
            "avg_sentence_length": round(float(avg_len), 4)
        }

    def _extract_document_text(self, file_path: str) -> str:
        """
        Extracts plain text from PDF or DOCX files.
        """
        ext = file_path.lower().split(".")[-1]
        text = ""
        try:
            if ext == "pdf":
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_path)
                for page in pdf:
                    text += page.get_textpage().get_text_range() + "\n"
            elif ext in ["docx", "doc"]:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception as e:
            text = f"Error extracting text: {str(e)}"
        
        return text.strip()

    def _scrape_url(self, url: str) -> str:
        """
        Scrapes article body text from a URL.
        """
        try:
            headers = {"User-Agent": "MenaCraft-NewsAnalyst/1.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove scripts, styles
            for s in soup(["script", "style"]): s.decompose()
            
            # Try to find common article tags
            article = soup.find("article") or soup.find("main") or soup
            text = article.get_text(separator=" ", strip=True)
            
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            return text[:5000] # Cap for safety
        except Exception as e:
            return f"Scraping failed: {str(e)}"