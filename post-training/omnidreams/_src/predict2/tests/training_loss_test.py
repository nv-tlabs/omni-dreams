# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Test that verifies specific training output patterns are present in the logs.
This test runs a training command and checks for expected iteration and loss patterns.

Usage:
    * [run all tests]: pytest -s omnidreams/_src/predict2/tests/training_loss_test.py --L1 --all 2>&1 | tee /tmp/err.log
"""

import re
import subprocess

import pytest

from omnidreams._src.imaginaire.utils.helper_test import RunIf


@RunIf(min_gpus=1)
@pytest.mark.L1
def test_training_output_patterns():
    """Test that verifies specific training output patterns are present in the logs."""

    # Define the command to run
    cmd = "torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=omnidreams/_src/predict2/configs/text2world/config.py -- experiment=error-free_ddp_mock-data_base-cb trainer.max_iter=3 trainer.cudnn.deterministic=True trainer.cudnn.benchmark=False"

    # Define the expected patterns to check for
    expected_patterns = [
        r"Iteration 1: Hit counter: 1/5 \| Loss: 16\.7822",
        r"Iteration 2: Hit counter: 2/5 \| Loss: 13\.3350",
        r"Iteration 3: Hit counter: 3/5 \| Loss: 17\.5436",
    ]

    # Run the command and capture output
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 minute timeout
        )

        # Get the combined output (stdout + stderr)
        output = result.stdout + result.stderr

        # Check if command was successful
        if result.returncode != 0:
            pytest.fail(f"Command failed with return code {result.returncode}. Error: {result.stderr}")

        # Check for each expected pattern
        missing_patterns = []
        for i, pattern in enumerate(expected_patterns, 1):
            if not re.search(pattern, output):
                missing_patterns.append(f"Pattern {i}: {pattern}")

        # If any patterns are missing, fail the test
        if missing_patterns:
            pytest.fail(
                "Missing expected patterns in output:\n"
                + "\n".join(missing_patterns)
                + f"\n\nCommand output:\n{output}"
            )

        # If we reach here, all patterns were found
        print("✓ All expected patterns found in training output")

    except subprocess.TimeoutExpired:
        pytest.fail(f"Command timed out after 5 minutes: {cmd}")
    except Exception as e:
        pytest.fail(f"Unexpected error running command: {e}")
