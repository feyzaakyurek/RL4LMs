from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from myutil import get_generations_gpt3, ForkedPdb
from custom_metric import EditMatchMetric
from typing import Dict, Any, List
import gem_metrics
from transformers import AutoTokenizer
import random
import re
from numpy import mean

CACHE_PATH = "/home/rl4lms/rl4lms/cache"

def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


class EditMatch(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.tokenizer_config = kwargs["tokenizer"]
        self.tokenizer = build_tokenizer(self.tokenizer_config)
        self.metric = EditMatchMetric(**kwargs["metric"])
        self.cache = {}


    def __call__(
        self,
        prev_observation: Observation,
        action: int,
        current_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        global calls

        if done:
            # 1. goal, steps, EOS, feedback_pred = Decode current_observation.input_encoded_pt
            # 2. edit_pred = query smallf (GPT-3)
            # 3. edit_gold = target_or_reference_texts
            # 4. reward = metric(edit_pred, edit_gold)

            state = current_observation.input_encoded_pt
            input_wfeed = self.tokenizer.decode(state[0], skip_special_tokens=True)

            # Get prompt and feedback separately.
            prompt_or_input_text = prev_observation.prompt_or_input_text
            feedback_pred = input_wfeed.lstrip(prompt_or_input_text)
            prompt_or_input_text = prompt_or_input_text.lstrip("Critique: ")
            edit_gold = current_observation.target_or_reference_texts
            metric_dict = self.metric.compute(
                prompt_texts = [prompt_or_input_text],
                generated_texts=[feedback_pred],
                reference_texts=[edit_gold]
            )
            reward = metric_dict[f"custom_metrics/editmatch_{self.metric.downstream_metric_name}"][-1]
            return reward

        return 0


# TODO: can we do batched?
# TODO num of environs = 10
# (can this be a list to track
# rouge on feedback_pred vs feedback_gold and edit_pred vs edit_gold?)
