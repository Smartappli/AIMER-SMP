from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union, Callable, Dict, List, Any

app = FastAPI()


class UnetParams(BaseModel):
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_channels: List[int] = (256, 128, 64, 32, 16) # same size than encoder_depth
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
    decoder_channels: List[int] = (256, 128, 64, 32, 16) # same size than encoder_depth
    decoder_use_batchnorm: bool = True
    decoder_attention_type: str | None = None
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    aux_params: dict | None = None


class FpnParams(BaseModel):
    encoder_name: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_pyramid_channels=256, decoder_segmentation_channels=128, decoder_merge_policy='add', decoder_dropout=0.2, in_channels=3, classes=1, activation=None, upsampling=4, aux_params=None


class MAnetParams(BaseModel):
    encoder_name=: str = "resnet34"
    encoder_depth: int = 5
    encoder_weights str | None = "imagenet"
    decoder_use_batchnorm: bool = True
    decoder_channels: List[int] = (256, 128, 64, 32, 16)
    decoder_pab_channels: int = 64
    in_channels=3
    classes=1
    activation: str | callable | None = None
    aux_params: dict | None = None


class PamParams(BaseModel):
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    encoder_output_stride: int = 16
    decoder_channels: int = 32
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    upsampling: int = 4
    aux_params: dict | None = None
