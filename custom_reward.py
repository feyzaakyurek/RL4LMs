from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from myutil import get_generations_gpt3
from typing import Dict, Any
import sys
import pdb
from transformers import AutoTokenizer
import random


tokenizer_config = {
    "model_name": "t5-base",
    "padding_side": "left",
    "truncation_side": "left",
    "pad_token_as_eos_token": False,
}


def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def exact_match(a, b):
    return a == b


class EditMatch(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.tokenizer_config = kwargs["tokenizer"]
        self.tokenizer = build_tokenizer(self.tokenizer_config)
        self.metric = exact_match
        self.prompt = kwargs["prompt_path"]
        self.separator = kwargs["separator"]

        # Load prompt.
        with open(self.prompt, "r") as f:
            self.prompt = f.read()

    def __call__(
        self,
        prev_observation: Observation,
        action: int,
        current_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

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

            # Prepend prompt.
            input_wfeed = (
                self.prompt
                + self.separator
                + prompt_or_input_text
                + "\nFeedback:"
                + feedback_pred
                + "\nEdit:"
            )

            # Here with some probability we append one of "Reorder" or "Insert"
            # to the input_wfeed.
            prob = random.uniform(0, 1)
            if prob < 0.25:
                append = " Reorder"
            elif prob < 5:
                append = " Insert"
            else:
                append = ""

            # Query GPT-3
            edit_pred = get_generations_gpt3(
                ls=[input_wfeed],
                model_name="code-davinci-002",
                clean_tok=True,
                stop=[self.separator],
                temperature=0.0,
                batch_size=20,
                max_length=50,
                penalty=0.7,
                n=1,
                keyfile="openai_key_me",
            )
            edit_pred = edit_pred[0] + append
            edit_pred = edit_pred.strip()  # Strip whitespace.

            # Reward
            edit_gold = current_observation.target_or_reference_texts[0]
            reward = self.metric(edit_pred, edit_gold)
            print("\n\nedit_pred:\n\n{}\t{}".format(edit_pred, reward))
            return reward + 0.0

        return 0


# TODO: can we do batched?
# TODO num of environs = 10
# (can this be a list to track
# rouge on feedback_pred vs feedback_gold and edit_pred vs edit_gold?)
