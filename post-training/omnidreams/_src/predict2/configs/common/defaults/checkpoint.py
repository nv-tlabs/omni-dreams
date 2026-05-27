# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from hydra.core.config_store import ConfigStore

from omnidreams._src.imaginaire.config import CheckpointConfig, ObjectStoreConfig

pbss_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/pbss_checkpoint.secret",
    bucket="checkpoints",
)
s3_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/s3_checkpoint.secret",
    bucket="bucket",
)
gcp_object_store = ObjectStoreConfig(
    enabled=False,
    credentials="credentials/gcp_checkpoint.secret",
    bucket="bucket",
)


CHECKPOINT_PBSS = CheckpointConfig(
    save_to_object_store=pbss_object_store,
    save_iter=1000,
    load_from_object_store=pbss_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)

CHECKPOINT_S3 = CheckpointConfig(
    save_to_object_store=s3_object_store,
    save_iter=1000,
    load_from_object_store=s3_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)

CHECKPOINT_GCP = CheckpointConfig(
    save_to_object_store=gcp_object_store,
    save_iter=1000,
    load_from_object_store=gcp_object_store,
    load_path="",
    load_training_state=False,
    strict_resume=True,
)


def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(group="checkpoint", package="checkpoint", name="pbss", node=CHECKPOINT_PBSS)
    cs.store(group="checkpoint", package="checkpoint", name="s3", node=CHECKPOINT_S3)
    cs.store(group="checkpoint", package="checkpoint", name="gcp", node=CHECKPOINT_GCP)
