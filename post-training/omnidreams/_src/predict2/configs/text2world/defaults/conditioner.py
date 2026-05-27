# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.lazy_config import LazyCall as L
from omnidreams._src.imaginaire.lazy_config import LazyDict
from omnidreams._src.predict2.conditioner import ReMapkey, TextAttr, TextAttrEmptyStringDrop, VideoConditioner

VideoConditionerFpsPaddingConfig: LazyDict = L(VideoConditioner)(
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
)


VideoConditionerFpsPaddingEmptyStringDrppConfig: LazyDict = L(VideoConditioner)(
    text=L(TextAttrEmptyStringDrop)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="add_fps_padding_mask",
        node=VideoConditionerFpsPaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="add_fps_padding_mask_empty_string_drop",
        node=VideoConditionerFpsPaddingEmptyStringDrppConfig,
    )
