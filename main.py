from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union, Callable, Dict, List, Any

app = FastAPI()


class UnetParams(BaseModel):
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_channels: Optional(List[int]) = None
    decoder_use_batchnorm: bool = True
    decoder_attention_type: str | None = None
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    aux_params: dict | None = None


class UnetPlusPlusParams(BaseModel):
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_channels: Optional(List[int]) = None
    
    
