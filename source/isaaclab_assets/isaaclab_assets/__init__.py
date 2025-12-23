# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os
import toml

from huggingface_hub import snapshot_download



# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]

CUSTOM_ASSETS_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "isaaclab_assets", "custom")
CUSTOM_ASSETS_PATH = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "isaaclab_assets", "custom", "assets")

if not os.path.exists(CUSTOM_ASSETS_PATH):
    snapshot_download(
        repo_id="isaacwilliam4/isaaclab-harl-dataset",
        repo_type="dataset",
        revision="main",
        local_dir=CUSTOM_ASSETS_DIR,
        allow_patterns=["assets/**"],
    )

from .robots import *
from .sensors import *
