"""
Model output contract for SatQuery AI evaluation.
Every model track (GeoChat, Qwen, Fusion) must emit results in THIS format.
"""
from pydantic import BaseModel
from typing import Optional, Literal

class ModelPrediction(BaseModel):
    sample_id: str
    task: Literal["vqa", "caption", "grounding", "change_vqa", "fusion_vqa"]
    
    # text outputs
    answer: Optional[str] = None          # vqa / change_vqa / caption
    
    # spatial outputs (grounding) — normalized [x_min, y_min, x_max, y_max], 0-1
    box: Optional[list[float]] = None
    mask_path: Optional[str] = None
    
    # metadata
    model_name: str
    runtime_sec: Optional[float] = None

class EvalResults(BaseModel):
    model_name: str
    benchmark: str                        # "vrsbench" | "rsvqa" | "cdvqa" | "probe"
    predictions: list[ModelPrediction]
