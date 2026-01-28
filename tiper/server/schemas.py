from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateJobRequest(BaseModel):
  source_png_path: str = Field(..., description="Path to the exported flatten PNG (Windows or WSL path).")
  config: Optional[Dict[str, Any]] = None


class BubbleBBox(BaseModel):
  left: float
  top: float
  right: float
  bottom: float


class Bubble(BaseModel):
  id: str
  bbox: BubbleBBox
  source: Optional[str] = None
  confidence: Optional[float] = None


class SubmitBubblesRequest(BaseModel):
  bubbles: List[Bubble]

