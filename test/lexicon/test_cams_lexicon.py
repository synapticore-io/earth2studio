# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from earth2studio.lexicon import CAMSLexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["dust"],
        ["so2sfc", "pm2p5"],
        ["no2sfc", "o3sfc", "cosfc"],
        ["aod550"],
        ["duaod550", "tcno2"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cams_lexicon(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = CAMSLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device


def test_cams_lexicon_invalid():
    with pytest.raises(KeyError):
        CAMSLexicon["nonexistent_variable"]


def test_cams_lexicon_vocab_format():
    for key, value in CAMSLexicon.VOCAB.items():
        parts = value.split("::")
        assert len(parts) == 4, f"VOCAB entry '{key}' must have format 'dataset::api_var::nc_key::level'"
        dataset, api_var, nc_key, level = parts
        assert dataset in (
            "cams-europe-air-quality-forecasts",
            "cams-global-atmospheric-composition-forecasts",
        ), f"Unknown dataset in VOCAB entry '{key}': {dataset}"
        assert api_var, f"VOCAB entry '{key}' has empty api_var"
        assert nc_key, f"VOCAB entry '{key}' has empty nc_key"
