from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union, Callable, Dict, List, Any

import segmentation_models_pytorch as smp


app = FastAPI()


class Encoder(str, Enum):
    SelecSls42 = "SelecSls42"
    SelecSls42b = "SelecSls42b"
    SelecSls60 = "SelecSls60"
    SelecSls60b = "SelecSls60b"
    SelecSls84 = "SelecSls84"
    bat_resnext26ts = "bat_resnext26ts"
    botnet26t_256 = "botnet26t_256"
    botnet50ts_256 = "botnet50ts_256"
    resnet34 = "resnet34"

    
class DeepLabV3Params(BaseModel):
    encoder_name: str = Encoder.resnet34
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_channels: int = 256
    in_channels: int = 3
    classes: int = 1
    activation: str | None = None
    upsampling: int = 8
    aux_params: dict | None = None


class DeepLabV3PlusParams(BaseModel):
    encoder_name: str = Encoder.resnet34
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    encoder_output_stride: int = 16
    decoder_channels: int = 256
    decoder_atrous_rates: tuple = (12, 24, 36)
    in_channels: int = 3
    classes: int = 1
    activation: str | None = None
    upsampling: int = 4
    aux_params: dict | None = None

    
class FpnParams(BaseModel):
    encoder_name: str = Encoder.resnet34
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_pyramid_channels: int = 256
    decoder_segmentation_channels: int = 128
    decoder_merge_policy: str = "add"
    decoder_dropout: float = 0.2
    in_channels: int = 3
    classes: int = 1
    activation: str | None = None
    upsampling: int = 4
    aux_params: dict | None = None


class MAnetParams(BaseModel):
    encoder_name=: str = Encoder.resnet34
    encoder_depth: int = 5
    encoder_weights str | None = "imagenet"
    decoder_use_batchnorm: bool = True
    decoder_channels: List[int] = (256, 128, 64, 32, 16)
    decoder_pab_channels: int = 64
    in_channels=3
    classes=1
    activation: str | callable | None = None
    aux_params: dict | None = None


class PanParams(BaseModel):
    encoder_name: str = Encoder.resnet34
    encoder_weights: str | None = "imagenet"
    encoder_output_stride: int = 16
    decoder_channels: int = 32
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    upsampling: int = 4
    aux_params: dict | None = None


class PspNetParams(BaseModel):
    encoder_name: str = Encoder.resnet34
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    psp_out_channels: int = 512
    psp_use_batchnor: bool = True
    psp_dropout: float = 0.2
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    upsampling: int = 4
    aux_params: dict | None = None


class UnetParams(BaseModel):
    encoder_name: str = Encoder.resnet34
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
    encoder_name: str = Encoder.resnet34
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_channels: List[int] = (256, 128, 64, 32, 16) # same size than encoder_depth
    decoder_use_batchnorm: bool = True
    decoder_attention_type: str | None = None
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    aux_params: dict | None = None


@app.get("/")
async def root():
    return {"pycaret_version": smp.__version__}

