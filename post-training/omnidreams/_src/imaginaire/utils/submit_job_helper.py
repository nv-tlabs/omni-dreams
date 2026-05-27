# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import os.path as osp

import git
from loguru import logger as logging


def is_git(path):
    try:
        _ = git.Repo(path, search_parent_directories=True).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


def git_rootdir(path=""):
    if is_git(os.getcwd()):
        git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
        root = git_repo.git.rev_parse("--show-toplevel")
        return osp.join(root, path)
    logging.info("not a git repo")
    return osp.join(os.getcwd(), path)
