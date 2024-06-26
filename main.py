from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union, Callable, Dict, List, Any

import segmentation_models_pytorch as smp


app = FastAPI()


class Encoder(str, Enum):
    selecsls42 = "SelecSls42"
    selecsls42b = "SelecSls42b"
    selecsls60 = "SelecSls60"
    selecsls60b = "SelecSls60b"
    selecsls84 = "SelecSls84"
    bat_resnext26ts = "bat_resnext26ts"
    botnet26t_256 = "botnet26t_256"
    botnet50ts_256 = "botnet50ts_256"
    coatnet_0_224 = "coatnet_0_224"
    coatnet_0_rw_224 = "coatnet_0_rw_224"
    coatnet_1_224 = "coatnet_1_224"
    coatnet_1_rw_224 = "coatnet_1_rw_224"
    coatnet_2_224 = "coatnet_2_224"
    coatnet_2_rw_224 = "coatnet_2_rw_224"
    coatnet_3_224 = "coatnet_3_224"
    coatnet_3_rw_224 = "coatnet_3_rw_224"
    coatnet_4_224 = "coatnet_4_224"
    coatnet_5_224 = "coatnet_5_224"
    coatnet_bn_0_rw_224 = "coatnet_bn_0_rw_224"
    coatnet_nano_cc_224 = "coatnet_nano_cc_224"
    coatnet_nano_rw_224 = "coatnet_nano_rw_224"
    coatnet_pico_rw_224 = "coatnet_pico_rw_224"
    coatnet_rmlp_0_rw_224 = "coatnet_rmlp_0_rw_224"
    coatnet_rmlp_1_rw2_224 = "coatnet_rmlp_1_rw2_224"
    coatnet_rmlp_1_rw_224 = "coatnet_rmlp_1_rw_224"
    coatnet_rmlp_2_rw_224 = "coatnet_rmlp_2_rw_224"
    coatnet_rmlp_2_rw_384 = "coatnet_rmlp_2_rw_384"
    coatnet_rmlp_3_rw_224 = "coatnet_rmlp_3_rw_224"
    coatnet_rmlp_nano_rw_224 = "coatnet_rmlp_nano_rw_224"
    coatnext_nano_rw_224 = "coatnext_nano_rw_224"
    cs3darknet_focus_l = "cs3darknet_focus_l"
    cs3darknet_focus_m = "cs3darknet_focus_m"
    cs3darknet_focus_s = "cs3darknet_focus_s"
    cs3darknet_focus_x = "cs3darknet_focus_x"
    cs3darknet_l = "cs3darknet_l"
    cs3darknet_m = "cs3darknet_m"
    cs3darknet_s = "cs3darknet_s"
    cs3darknet_x = "cs3darknet_x"
    cs3edgenet_x = "cs3edgenet_x"
    cs3se_edgenet_x = "cs3se_edgenet_x"
    cs3sedarknet_l = "cs3sedarknet_l"
    cs3sedarknet_x = "cs3sedarknet_x"
    cs3sedarknet_xdw = "cs3sedarknet_xdw"
    cspresnet50 = "cspresnet50"
    cspresnet50d = "cspresnet50d"
    cspresnet50w = "cspresnet50w"
    cspresnext50 = "cspresnext50"
    densenet121 = "densenet121"
    densenet161 = "densenet161"
    densenet169 = "densenet169"
    densenet201 = "densenet201"
    densenet264d = "densenet264d"
    densenetblur121d = "densenetblur121d"
    dla102 = "dla102"
    dla102x = "dla102x"
    dla102x2 = "dla102x2"
    dla169 = "dla169"
    dla34 = "dla34"
    dla46_c = "dla46_c"
    dla46x_c = "dla46x_c"
    dla60 = "dla60"
    dla60_res2net = "dla60_res2net"
    dla60_res2next = "dla60_res2next"
    dla60x = "dla60x"
    dla60x_c = "dla60x_c"
    dm_nfnet_f0 = "dm_nfnet_f0"
    dm_nfnet_f1 = "dm_nfnet_f1"
    dm_nfnet_f2 = "dm_nfnet_f2"
    dm_nfnet_f3 = "dm_nfnet_f3"
    dm_nfnet_f4 = "dm_nfnet_f4"
    dm_nfnet_f5 = "dm_nfnet_f5"
    dm_nfnet_f6 = "dm_nfnet_f6"
    dpn107 = "dpn107"
    dpn131 = "dpn131"
    dpn48b = "dpn48b"
    dpn68 = "dpn68"
    dpn68b = "dpn68b"
    dpn92 = "dpn92"
    dpn98 = "dpn98"
    eca_botnext26ts_256 = "eca_botnext26ts_256"
    eca_halonext26ts = "eca_halonext26ts"
    eca_nfnet_l0 = "eca_nfnet_l0"
    eca_nfnet_l1 = "eca_nfnet_l1"
    eca_nfnet_l2 = "eca_nfnet_l2"
    eca_nfnet_l3 = "eca_nfnet_l3"
    eca_resnet33ts = "eca_resnet33ts"
    eca_resnext26ts = "eca_resnext26ts"
    eca_vovnet39b = "eca_vovnet39b"
    ecaresnet101d = "ecaresnet101d"
    ecaresnet101d_pruned = "ecaresnet101d_pruned"
    ecaresnet200d = "ecaresnet200d"
    ecaresnet269d = "ecaresnet269d"
    ecaresnet26t = "ecaresnet26t"
    ecaresnet50d = "ecaresnet50d"
    ecaresnet50d_pruned = "ecaresnet50d_pruned"
    ecaresnet50t = "ecaresnet50t"
    ecaresnetlight = "ecaresnetlight"
    ecaresnext26t_32x4d = "ecaresnext26t_32x4d"
    ecaresnext50t_32x4d = "ecaresnext50t_32x4d"
    efficientnet_b0 = "efficientnet_b0"
    efficientnet_b0_g16_evos = "efficientnet_b0_g16_evos"
    efficientnet_b0_g8_gn = "efficientnet_b0_g8_gn"
    efficientnet_b0_gn = "efficientnet_b0_gn"
    efficientnet_b1 = "efficientnet_b1"
    efficientnet_b1_pruned = "efficientnet_b1_pruned"
    efficientnet_b2 = "efficientnet_b2"
    efficientnet_b2_pruned = "efficientnet_b2_pruned"
    efficientnet_b2a = "efficientnet_b2a"
    efficientnet_b3 = "efficientnet_b3"
    efficientnet_b3_g8_gn = "efficientnet_b3_g8_gn"
    efficientnet_b3_gn = "efficientnet_b3_gn"
    efficientnet_b3_pruned = "efficientnet_b3_pruned"
    efficientnet_b3a = "efficientnet_b3a"
    efficientnet_b4 = "efficientnet_b4"
    efficientnet_b5 = "efficientnet_b5"
    efficientnet_b6 = "efficientnet_b6"
    efficientnet_b7 = "efficientnet_b7"
    efficientnet_b8 = "efficientnet_b8"
    efficientnet_cc_b0_4e = "efficientnet_cc_b0_4e"
    efficientnet_cc_b0_8e = "efficientnet_cc_b0_8e"
    efficientnet_cc_b1_8e = "efficientnet_cc_b1_8e"
    efficientnet_el = "efficientnet_el"
    efficientnet_el_pruned = "efficientnet_el_pruned"
    efficientnet_em = "efficientnet_em"
    efficientnet_es + "efficientnet_es"
    efficientnet_es_pruned = "efficientnet_es_pruned"
    efficientnet_l2 = "efficientnet_l2"
    efficientnet_lite0 = "efficientnet_lite0"
    efficientnet_lite1 = "efficientnet_lite1"
    efficientnet_lite2 = "efficientnet_lite2"
    efficientnet_lite3 = "efficientnet_lite3"
    efficientnet_lite4 = "efficientnet_lite4"
    efficientnetv2_l = "efficientnetv2_l"
    efficientnetv2_m = "efficientnetv2_m"
    efficientnetv2_rw_m + "efficientnetv2_rw_m"
    efficientnetv2_rw_s = "efficientnetv2_rw_s"
    efficientnetv2_rw_t = "efficientnetv2_rw_t"
    efficientnetv2_s = "efficientnetv2_s"
    efficientnetv2_xl = "efficientnetv2_xl"
    ese_vovnet19b_dw = "ese_vovnet19b_dw"
    ese_vovnet19b_slim = "ese_vovnet19b_slim"
    ese_vovnet19b_slim_dw = "ese_vovnet19b_slim_dw"
    ese_vovnet39b = "ese_vovnet39b"
    ese_vovnet39b_evos = "ese_vovnet39b_evos"
    ese_vovnet57b = "ese_vovnet57b"
    ese_vovnet99b = "ese_vovnet99b"
    fbnetc_100 = "fbnetc_100"
    fbnetv3_b = "fbnetv3_b"
    fbnetv3_d = "fbnetv3_d"
    fbnetv3_g = "fbnetv3_g"
    gc_efficientnetv2_rw_t = "gc_efficientnetv2_rw_t"
    gcresnet33ts = "gcresnet33ts"
    gcresnet50t = "gcresnet50t"
    gcresnext26ts = "gcresnext26ts"
    gcresnext50ts = "gcresnext50ts"
    gernet_l = "gernet_l"
    gernet_m = "gernet_m"
    gernet_s = "gernet_s"
    ghostnet_050 = "ghostnet_050"
    ghostnet_100 = "ghostnet_100"
    ghostnet_130 = "ghostnet_130"
    halo2botnet50ts_256 = "halo2botnet50ts_256"
    halonet26t = "halonet26t"
    halonet50ts = "halonet50ts"
    halonet_h1 = "halonet_h1"
    haloregnetz_b = "haloregnetz_b"
    hardcorenas_a = "hardcorenas_a"
    hardcorenas_b = "hardcorenas_b"
    hardcorenas_c = "hardcorenas_c"
    hardcorenas_d = "hardcorenas_d"
    hardcorenas_e = "hardcorenas_e"
    hardcorenas_f = "hardcorenas_f"
    hrnet_w18 = "hrnet_w18"
    hrnet_w18_small = "hrnet_w18_small"
    hrnet_w18_small_v2 = "hrnet_w18_small_v2"
    hrnet_w18_ssld = "hrnet_w18_ssld"
    hrnet_w30 = "hrnet_w30"
    hrnet_w32 = "hrnet_w32"
    hrnet_w40 = "hrnet_w40"
    hrnet_w44 + "hrnet_w44"
    hrnet_w48 = "hrnet_w48"
    hrnet_w48_ssld = "hrnet_w48_ssld"
    hrnet_w64 = "hrnet_w64"
    inception_resnet_v2 = "inception_resnet_v2"
    inception_v3 = "inception_v3"
    inception_v4 = "inception_v4"
    lambda_resnet26rpt_256 = "lambda_resnet26rpt_256"
    lambda_resnet26t = "lambda_resnet26t"
    lambda_resnet50ts = "lambda_resnet50ts"
    lamhalobotnet50ts_256 = "lamhalobotnet50ts_256"
    lcnet_035 = "lcnet_035"
    lcnet_050 = "lcnet_050"
    lcnet_075 = "lcnet_075"
    lcnet_100 = "lcnet_100"
    lcnet_150 = "lcnet_150"
    legacy_senet154 = "legacy_senet154"
    legacy_seresnet101 = "legacy_seresnet101"
    legacy_seresnet152 = "legacy_seresnet152"
    legacy_seresnet18 = "legacy_seresnet18"
    legacy_seresnet34 = "legacy_seresnet34"
    legacy_seresnet50 = "legacy_seresnet50"
    legacy_seresnext101_32x4d = "legacy_seresnext101_32x4d"
    legacy_seresnext26_32x4d = "legacy_seresnext26_32x4d"
    legacy_seresnext50_32x4d = "legacy_seresnext50_32x4d"
    legacy_xception = "legacy_xception"
    maxvit_base_tf_224 = "maxvit_base_tf_224"
    maxvit_base_tf_384 = "maxvit_base_tf_384"
    maxvit_base_tf_512 = "maxvit_base_tf_512"
    maxvit_large_tf_224 = "maxvit_large_tf_224"
    maxvit_large_tf_384 = "maxvit_large_tf_384"
    maxvit_large_tf_512 = "maxvit_large_tf_512"
    maxvit_nano_rw_256 = "maxvit_nano_rw_256"
    maxvit_pico_rw_256 = "maxvit_pico_rw_256"
    maxvit_rmlp_base_rw_224 = "maxvit_rmlp_base_rw_224"
    maxvit_rmlp_base_rw_384 = "maxvit_rmlp_base_rw_384"
    maxvit_rmlp_nano_rw_256 = "maxvit_rmlp_nano_rw_256"
    maxvit_rmlp_pico_rw_256 = "maxvit_rmlp_pico_rw_256"
    maxvit_rmlp_small_rw_224 = "maxvit_rmlp_small_rw_224"
    maxvit_rmlp_small_rw_256 = "maxvit_rmlp_small_rw_256"
    maxvit_rmlp_tiny_rw_256 = "maxvit_rmlp_tiny_rw_256"
    maxvit_small_tf_224 = "maxvit_small_tf_224"
    maxvit_small_tf_384 = "maxvit_small_tf_384"
    maxvit_small_tf_512 = "maxvit_small_tf_512"
    maxvit_tiny_pm_256 = "maxvit_tiny_pm_256"
    maxvit_tiny_rw_224 = "maxvit_tiny_rw_224"
    maxvit_tiny_rw_256 = "maxvit_tiny_rw_256"
    maxvit_tiny_tf_224 = "maxvit_tiny_tf_224"
    maxvit_tiny_tf_384 = "maxvit_tiny_tf_384"
    maxvit_tiny_tf_512 = "maxvit_tiny_tf_512"
    maxvit_xlarge_tf_224 = "maxvit_xlarge_tf_224"
    maxvit_xlarge_tf_384 = "maxvit_xlarge_tf_384"
    maxvit_xlarge_tf_512 = "maxvit_xlarge_tf_512"
    maxxvit_rmlp_nano_rw_256 = "maxxvit_rmlp_nano_rw_256"
    maxxvit_rmlp_small_rw_256 = "maxxvit_rmlp_small_rw_256"
    maxxvit_rmlp_tiny_rw_256 = "maxxvit_rmlp_tiny_rw_256"
    maxxvitv2_nano_rw_256 = "maxxvitv2_nano_rw_256"
    maxxvitv2_rmlp_base_rw_224 = "maxxvitv2_rmlp_base_rw_224"
    maxxvitv2_rmlp_base_rw_384 = "maxxvitv2_rmlp_base_rw_384"
    maxxvitv2_rmlp_large_rw_224 = "maxxvitv2_rmlp_large_rw_224"
    mixnet_l = "mixnet_l"
    mixnet_m = "mixnet_m"
    mixnet_s = "mixnet_s"
    mixnet_xl = "mixnet_xl"
    mixnet_xxl = "mixnet_xxl"
    mnasnet_050 = "mnasnet_050"
    mnasnet_075 = "mnasnet_075"
    mnasnet_100 = "mnasnet_100"
    mnasnet_140 = "mnasnet_140"
    mnasnet_a1 = "mnasnet_a1"
    mnasnet_b1 = "mnasnet_b1"
    mnasnet_small = "mnasnet_small"
    mobilenetv2_035 = "mobilenetv2_035"
    mobilenetv2_050 = "mobilenetv2_050"
    mobilenetv2_075 = "mobilenetv2_075"
    mobilenetv2_100 = "mobilenetv2_100"
    mobilenetv2_110d = "mobilenetv2_110d"
    mobilenetv2_120d = "mobilenetv2_120d"
    mobilenetv2_140 = "mobilenetv2_140"
    mobilenetv3_large_075 = "mobilenetv3_large_075"
    mobilenetv3_large_100 = "mobilenetv3_large_100"
    mobilenetv3_rw = "mobilenetv3_rw"
    mobilenetv3_small_050 = "mobilenetv3_small_050"
    mobilenetv3_small_075 = "mobilenetv3_small_075"
    mobilenetv3_small_100 = "mobilenetv3_small_100"
    mobilevit_s = "mobilevit_s"
    mobilevit_xs = "mobilevit_xs"
    mobilevit_xxs = "mobilevit_xxs"
    mobilevitv2_050 = "mobilevitv2_050"
    mobilevitv2_075 = "mobilevitv2_075"
    mobilevitv2_100 = "mobilevitv2_100"
    mobilevitv2_125 = "mobilevitv2_125"
    mobilevitv2_150 = "mobilevitv2_150"
    mobilevitv2_175 = "mobilevitv2_175"
    mobilevitv2_200 = "mobilevitv2_200"
    nasnetalarge = "nasnetalarge"
    nf_ecaresnet101 = "nf_ecaresnet101"
    nf_ecaresnet26 = "nf_ecaresnet26"
    nf_ecaresnet50 = "nf_ecaresnet50"
    nf_regnet_b0 = "nf_regnet_b0"
    nf_regnet_b1 = "nf_regnet_b1"
    nf_regnet_b2 = "nf_regnet_b2"
    nf_regnet_b3 = "nf_regnet_b3"
    nf_regnet_b4 = "nf_regnet_b4"
    nf_regnet_b5 = "nf_regnet_b5"
    nf_resnet101 = "nf_resnet101"
    nf_resnet26 = "nf_resnet26"
    nf_resnet50 = "nf_resnet50"
    nf_seresnet101 = "nf_seresnet101"
    nf_seresnet26 = "nf_seresnet26"
    nf_seresnet50 = "nf_seresnet50"
    nfnet_f0 = "nfnet_f0"
    nfnet_f1 = "nfnet_f1"
    nfnet_f2 = "nfnet_f2"
    nfnet_f3 = "nfnet_f3"
    nfnet_f4 = "nfnet_f4"
    nfnet_f5 = "nfnet_f5"
    nfnet_f6 = "nfnet_f6"
    nfnet_f7 = "nfnet_f7"
    nfnet_l0 = "nfnet_l0"
    pnasnet5large = "pnasnet5large"
    regnetv_040 = "regnetv_040"
    regnetv_064 = "regnetv_064"
    regnetx_002 = "regnetx_002"
    regnetx_004 = "regnetx_004"
    regnetx_004_tv = "regnetx_004_tv"
    regnetx_006 = "regnetx_006"
    regnetx_008 = "regnetx_008"
    regnetx_016 = "regnetx_016"
    regnetx_032 = "regnetx_032"
    regnetx_040 = "regnetx_040"
    regnetx_064 = "regnetx_064"
    regnetx_080 = "regnetx_080"
    regnetx_120 = "regnetx_120"
    regnetx_160 = "regnetx_160"
    regnetx_320 = "regnetx_320"
    regnety_002 = "regnety_002"
    regnety_004 = "regnety_004"
    regnety_006 = "regnety_006"
    regnety_008 = "regnety_008"
    regnety_008_tv = "regnety_008_tv"
    regnety_016 = "regnety_016"
    regnety_032 = "regnety_032"
    regnety_040 = "regnety_040"
    regnety_040_sgn = "regnety_040_sgn"
    regnety_064 = "regnety_064"
    regnety_080 = "regnety_080"
    regnety_080_tv = "regnety_080_tv"
    regnety_120 = "regnety_120"
    regnety_1280 = "regnety_1280"
    regnety_160 = "regnety_160"
    regnety_2560 = "regnety_2560"
    regnety_320 = "regnety_320"
    regnety_640 = "regnety_640"
    regnetz_005 = "regnetz_005"
    regnetz_040 = "regnetz_040"
    regnetz_040_h = "regnetz_040_h"
    regnetz_b16 = "regnetz_b16"
    regnetz_b16_evos = "regnetz_b16_evos"
    regnetz_c16 = "regnetz_c16"
    regnetz_c16_evos = "regnetz_c16_evos"
    regnetz_d32 = "regnetz_d32"
    regnetz_d8 = "regnetz_d8"
    regnetz_d8_evos = "regnetz_d8_evos"
    regnetz_e8 = "regnetz_e8"
    repvgg_a2 = "repvgg_a2"
    repvgg_b0 = "repvgg_b0"
    repvgg_b1 = "repvgg_b1"
    repvgg_b1g4 = "repvgg_b1g4"
    repvgg_b2 = "repvgg_b2"
    repvgg_b2g4 = "repvgg_b2g4"
    repvgg_b3 = "repvgg_b3"
    repvgg_b3g4 = "repvgg_b3g4"
    res2net101_26w_4s = "res2net101_26w_4s"
    res2net101d = "res2net101d"
    res2net50_14w_8s = "res2net50_14w_8s"
    res2net50_26w_4s = "res2net50_26w_4s"
    res2net50_26w_6s = "res2net50_26w_6s"
    res2net50_26w_8s = "res2net50_26w_8s"
    res2net50_48w_2s = "res2net50_48w_2s"
    res2net50d = "res2net50d"
    res2next50 = "res2next50"
    resnest101e = "resnest101e"
    resnest14d = "resnest14d"
    resnest200e = "resnest200e"
    resnest269e = "resnest269e"
    resnest26d = "resnest26d"
    resnest50d = "resnest50d"
    resnest50d_1s4x24d = "resnest50d_1s4x24d"
    resnest50d_4s2x40d = "resnest50d_4s2x40d"
    resnet101 = "resnet101"
    resnet101c = "resnet101c"
    resnet101d = "resnet101d"
    resnet101s = "resnet101s"
    resnet10t = "resnet10t"
    resnet14t = "resnet14t"
    resnet152 = "resnet152"
    resnet152c = "resnet152c"
    resnet152d = "resnet152d"
    resnet152s = "resnet152s"
    resnet18 = "resnet18"
    resnet18d = "resnet18d"
    resnet200 = "resnet200"
    resnet200d = "resnet200d"
    resnet26 = "resnet26"
    resnet26d = "resnet26d"
    resnet26t = "resnet26t"
    resnet32ts = "resnet32ts"
    resnet33ts = "resnet33ts"
    resnet34 = "resnet34"
    resnet34d = "resnet34d"
    resnet50 = "resnet50"
    resnet50_gn = "resnet50_gn"
    resnet50c = "resnet50c"
    resnet50d = "resnet50d"
    resnet50s = "resnet50s"
    resnet50t = "resnet50t"
    resnet51q = "resnet51q"
    resnet61q = "resnet61q"
    resnetaa101d = "resnetaa101d"
    resnetaa34d = "resnetaa34d"
    resnetaa50 = "resnetaa50"
    resnetaa50d = "resnetaa50d"
    resnetblur101d = "resnetblur101d"
    resnetblur18 = "resnetblur18"
    resnetblur50 = "resnetblur50"
    resnetblur50d = "resnetblur50d"
    resnetrs101 = "resnetrs101"
    resnetrs152 = "resnetrs152"
    resnetrs200 = "resnetrs200"
    resnetrs270 = "resnetrs270"
    resnetrs350 = "resnetrs350"
    resnetrs420 = "resnetrs420"
    resnetrs50 = "resnetrs50"
    resnetv2_101 = "resnetv2_101"
    resnetv2_101d = "resnetv2_101d"
    resnetv2_101x1_bit = "resnetv2_101x1_bit"
    resnetv2_101x3_bit = "resnetv2_101x3_bit"
    resnetv2_152 = "resnetv2_152"
    resnetv2_152d = "resnetv2_152d"
    resnetv2_152x2_bit = "resnetv2_152x2_bit"
    resnetv2_152x4_bit = "resnetv2_152x4_bit"
    resnetv2_50 = "resnetv2_50"
    resnetv2_50d = "resnetv2_50d"
    resnetv2_50d_evos = "resnetv2_50d_evos"
    resnetv2_50d_frn = "resnetv2_50d_frn"
    resnetv2_50d_gn = "resnetv2_50d_gn"
    resnetv2_50t = "resnetv2_50t"
    resnetv2_50x1_bit = "resnetv2_50x1_bit"
    resnetv2_50x3_bit = "resnetv2_50x3_bit"
    resnext101_32x16d = "resnext101_32x16d"
    resnext101_32x32d = "resnext101_32x32d"
    resnext101_32x4d = "resnext101_32x4d"
    resnext101_32x8d = "resnext101_32x8d"
    resnext101_64x4d = "resnext101_64x4d"
    resnext26ts = "resnext26ts"
    resnext50_32x4d = "resnext50_32x4d"
    resnext50d_32x4d = "resnext50d_32x4d"
    rexnet_100 = "rexnet_100"
    rexnet_130 = "rexnet_130"
    rexnet_150 = "rexnet_150"
    rexnet_200 = "rexnet_200"
    rexnet_300 = "rexnet_300"
    rexnetr_100 = "rexnetr_100"
    rexnetr_130 = "rexnetr_130"
    rexnetr_150 = "rexnetr_150"
    rexnetr_200 = "rexnetr_200"
    rexnetr_300 = "rexnetr_300"
    sebotnet33ts_256 = "sebotnet33ts_256"
    sehalonet33ts = "sehalonet33ts"
    semnasnet_050 = "semnasnet_050"
    semnasnet_075 = "semnasnet_075"
    semnasnet_100 = "semnasnet_100"
    semnasnet_140 = "semnasnet_140"
    senet154 = "senet154"
    seresnet101 = "seresnet101"
    seresnet152 = "seresnet152"
    seresnet152d = "seresnet152d"
    seresnet18 = "seresnet18"
    seresnet200d = "seresnet200d"
    seresnet269d = "seresnet269d"
    seresnet33ts = "seresnet33ts"
    seresnet34 = "seresnet34"
    seresnet50 = "seresnet50"
    seresnet50t = "seresnet50t"
    seresnetaa50d = "seresnetaa50d"
    seresnext101_32x4d = "seresnext101_32x4d"
    seresnext101_32x8d = "seresnext101_32x8d"
    seresnext101_64x4d = "seresnext101_64x4d"
    seresnext101d_32x8d = "seresnext101d_32x8d"
    seresnext26d_32x4d = "seresnext26d_32x4d"
    seresnext26t_32x4d = "seresnext26t_32x4d"
    seresnext26tn_32x4d = "seresnext26tn_32x4d"
    seresnext26ts = "seresnext26ts"
    seresnext50_32x4d = "seresnext50_32x4d"
    seresnextaa101d_32x8d = "seresnextaa101d_32x8d"
    skresnet18 = "skresnet18"
    skresnet34 = "skresnet34"
    skresnet50 = "skresnet50"
    skresnet50d = "skresnet50d"
    skresnext50_32x4d = "skresnext50_32x4d"
    spnasnet_100 = "spnasnet_100"
    tf_efficientnet_b0 = "tf_efficientnet_b0"
    tf_efficientnet_b1 = "tf_efficientnet_b1"
    tf_efficientnet_b2 = "tf_efficientnet_b2"
    tf_efficientnet_b3 = "tf_efficientnet_b3"
    tf_efficientnet_b4 = "tf_efficientnet_b4"
    tf_efficientnet_b5 = "tf_efficientnet_b5"
    tf_efficientnet_b6 = "tf_efficientnet_b6"
    tf_efficientnet_b7 = "tf_efficientnet_b7"
    tf_efficientnet_b8 = "tf_efficientnet_b8"
    tf_efficientnet_cc_b0_4e = "tf_efficientnet_cc_b0_4e"
    tf_efficientnet_cc_b0_8e = "tf_efficientnet_cc_b0_8e"
    tf_efficientnet_cc_b1_8e = "tf_efficientnet_cc_b1_8e"
    tf_efficientnet_el = "tf_efficientnet_el"
    tf_efficientnet_em = "tf_efficientnet_em"
    tf_efficientnet_es = "tf_efficientnet_es"
    tf_efficientnet_l2 = "tf_efficientnet_l2"
    tf_efficientnet_lite0 = "tf_efficientnet_lite0"
    tf_efficientnet_lite1 = "tf_efficientnet_lite1"
    tf_efficientnet_lite2 = "tf_efficientnet_lite2"
    tf_efficientnet_lite3 = "tf_efficientnet_lite3"
    tf_efficientnet_lite4 = "tf_efficientnet_lite4"
    tf_efficientnetv2_b0 = "tf_efficientnetv2_b0"
    tf_efficientnetv2_b1 = "tf_efficientnetv2_b1"
    tf_efficientnetv2_b2 = "tf_efficientnetv2_b2"
    tf_efficientnetv2_b3 = "tf_efficientnetv2_b3"
    tf_efficientnetv2_l = "tf_efficientnetv2_l"
    tf_efficientnetv2_m = "tf_efficientnetv2_m"
    tf_efficientnetv2_s = "tf_efficientnetv2_s"
    tf_efficientnetv2_xl = "tf_efficientnetv2_xl"
    tf_mixnet_l = "tf_mixnet_l"
    tf_mixnet_m = "tf_mixnet_m"
    tf_mixnet_s = "tf_mixnet_s"
    tf_mobilenetv3_large_075 = "tf_mobilenetv3_large_075"
    tf_mobilenetv3_large_100 = "tf_mobilenetv3_large_100"
    tf_mobilenetv3_large_minimal_100 = "tf_mobilenetv3_large_minimal_100"
    tf_mobilenetv3_small_075 = "tf_mobilenetv3_small_075"
    tf_mobilenetv3_small_100 = "tf_mobilenetv3_small_100"
    tf_mobilenetv3_small_minimal_100 = "tf_mobilenetv3_small_minimal_100"
    tinynet_a = "tinynet_a"
    tinynet_b = "tinynet_b"
    tinynet_c = "tinynet_c"
    tinynet_d = "tinynet_d"
    tinynet_e = "tinynet_e"
    vovnet39a = "vovnet39a"
    vovnet39a = "vovnet39a"
    wide_resnet101_2 = "wide_resnet101_2"
    wide_resnet50_2 = "wide_resnet50_2"
    xception41 = "xception41"
    xception41p = "xception41p"
    xception65 = "xception65"
    xception65p = "xception65p"
    xception71 = "xception71"


    class DeepLabV3Params(BaseModel):
    encoder_name: str = Encoder.resnet34.value
    encoder_depth: int = 5
    encoder_weights: str | None = "imagenet"
    decoder_channels: int = 256
    in_channels: int = 3
    classes: int = 1
    activation: str | None = None
    upsampling: int = 8
    aux_params: dict | None = None


class DeepLabV3PlusParams(BaseModel):
    encoder_name: str = Encoder.resnet34.value
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
    encoder_name: str = Encoder.resnet34.value
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
    encoder_name=: str = Encoder.resnet34.value
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
    encoder_name: str = Encoder.resnet34.value
    encoder_weights: str | None = "imagenet"
    encoder_output_stride: int = 16
    decoder_channels: int = 32
    in_channels: int = 3
    classes: int = 1
    activation: str | callable | None = None
    upsampling: int = 4
    aux_params: dict | None = None


class PspNetParams(BaseModel):
    encoder_name: str = Encoder.resnet34.value
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
    encoder_name: str = Encoder.resnet34.value
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
    encoder_name: str = Encoder.resnet34.value
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

