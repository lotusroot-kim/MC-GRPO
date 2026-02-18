# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
            "'cispo': Clips the importance sampling weights instead of the advantage scaled importance weights. "
            "The clipped weights are then multiplied with the advantages and policy model's log probs. "
            "Individual token losses are aggregated by normalizing with the number of active tokens in "
            "the global accumulated batch. This method was introduced in the "
            "[MiniMax-M1 paper](https://huggingface.co/papers/2506.13585)."
            "'sapo': Soft Adaptive Policy Optimization loss, as introduced in the "
            "[Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2506.13585). "
            "Replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates "
            "off-policy updates while preserving useful learning signals."
        },
    )


    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The name to store runs under.")},
    )
    metric: Optional[str] = field(
        default='smallest',
        metadata={"help": ("What metrics are used for pruning? smallest, largest or random.")},
    )
    vllm_group_port: int = field(
    default=51216,
    metadata={
        "help": "Port number for the weight update group. This is used to communicate with the vLLM server. "
        "Unless the port is occupied, there is no need to change it.",
    },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device to use for vLLM. If 'auto', automatically selects a device. "
            "Note: This parameter is deprecated in favor of vllm_mode. It is kept for backward compatibility."
        },
    )
    allocation: bool = field(
        default=False,
        metadata={"help": "Whether to use allocation for GSM training."},
    )
    pruning: float = field(
        default=0.0,
        metadata={"help": "Pruning rate for GSM training (0.0 to 1.0)."},
    )
    mc: bool = field(
        default=False,
        metadata={"help": "Use median instead of mean for advantages calculation (median completion)."},
    )
    signflip_percent: float = field(
        default=0.0,
        metadata={"help": "Probability of flipping the sign of each advantage (0.0 to 1.0). Used for ablation studies."},
    )
    sample_num: int = field(
        default=0,
        metadata={"help": "Number of samples to use from the dataset. If 0, use all samples."},
    )
    


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
