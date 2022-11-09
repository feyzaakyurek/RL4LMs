from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.metric import RougeMetric
from myutil import get_generations_gpt3
from typing import Dict, Any, List
import gem_metrics
import sys
import pdb
from transformers import AutoTokenizer
import random
import re
from numpy import mean

CACHE_PATH = "/home/rl4lms/rl4lms/cache"


tokenizer_config = {
    "model_name": "t5-base",
    "padding_side": "left",
    "truncation_side": "left",
    "pad_token_as_eos_token": False,
}


def rouge1_metric(pred: List[str], ref: List[List[str]]):
    res = RougeMetric().compute(
        prompt_texts=[], generated_texts=pred, reference_texts=ref
    )
    return res["lexical/rouge_rouge1"][-1]
    # res = {}
    # p = gem_metrics.texts.Predictions({"values": [pred]})
    # r = gem_metrics.texts.References({"values": [ref]})
    # res = gem_metrics.compute(p, r, metrics_list=["rouge"])
    # return res["rouge1"]["fmeasure"]


def rouge_combined(pred: List[str], ref: List[List[str]]):
    rouge_keys = ["rouge1", "rouge2", "rougeL"]
    res = RougeMetric().compute(
        prompt_texts=[], generated_texts=pred, reference_texts=ref
    )
    rouge_scores = [res["lexical/rouge_" + k][-1] for k in rouge_keys]
    return mean(rouge_scores)


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


def custom_metric(pred, gold):

    score = 0.2
    try:
        if "[INSERT]" in pred and "Insert" in gold:

            if "[AFTER]" in pred and "after" in gold:

                node_insert = (
                    re.search("\[INSERT\](.*)\[AFTER\]", pred).group(1).strip()
                )
                node_after = re.search("\[AFTER\](.*)\[END\]", pred).group(1).strip()
                node_insert_g = re.search("Insert node(.*)after", gold).group(1).strip()
                node_after_g = re.search("after(.*)", gold).group(1).strip()

                if node_insert == node_insert_g:
                    score += 0.4
                if node_after == node_after_g:
                    score += 0.4

            elif "[BEFORE]" in pred and "before" in gold:

                node_insert = (
                    re.search("\[INSERT\](.*)\[BEFORE\]", pred).group(1).strip()
                )
                node_before = re.search("\[BEFORE\](.*)\[END\]", pred).group(1).strip()
                node_insert_g = (
                    re.search("Insert node(.*)before", gold).group(1).strip()
                )
                node_before_g = re.search("before(.*)", gold).group(1).strip()

                if node_insert == node_insert_g:
                    score += 0.4
                if node_before == node_before_g:
                    score += 0.4

            else:
                return score

        elif "[REMOVE]" in pred and "Remove" in gold:

            node_remove = re.search("\[REMOVE\](.*)\[END\]", pred).group(1).strip()
            node_remove_g = re.search("Remove node(.*)", gold).group(1).strip()

            if node_remove == node_remove_g:
                score += 0.8

        elif "[REORDER]" in pred and "Reorder" in gold:
            node_reorder1 = (
                re.search("\[REORDER\](.*)\[AND\]", pred).group(1).strip().strip("'")
            )
            node_reorder2 = (
                re.search("\[AND\](.*)\[END\]", pred).group(1).strip().strip("'")
            )

            node_reorder1_g = (
                re.search("Reorder edge between '<(.*),", gold).group(1).strip()
            )
            node_reorder2_g = re.search(",(.*) >'", gold).group(1).strip()

            ss = set([node_reorder1, node_reorder1_g, node_reorder2, node_reorder2_g])
            if len(ss) == 2:
                score += 0.8
            if len(ss) == 3:
                score += 0.4

        else:
            return 0.0
    except AttributeError:
        return score
    return score


metric_map = {
    "custom": custom_metric,
    "rouge1": rouge1_metric,
    "rouge_combined": rouge_combined,
}


class EditMatch(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.tokenizer_config = kwargs["tokenizer"]
        self.tokenizer = build_tokenizer(self.tokenizer_config)
        self.metric_name = kwargs["metric"]
        self.metric = metric_map[self.metric_name]
        self.prompt = kwargs["prompt_path"]
        self.separator = kwargs["separator"]
        self.openai_api_key = kwargs["openai_key"]
        self.model_name = kwargs["gpt3_model_name"]
        self.cache = {}

        # Load prompt.
        with open(self.prompt, "r") as f:
            self.prompt = f.read()

        # Check key is valid.
        if self.model_name != "code-davinci-002":
            assert self.openai_api_key == "openai_key_ai2"

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

            # Prepend prompt.
            input_wfeed = (
                self.prompt
                + self.separator
                + prompt_or_input_text
                + "\nFeedback:"
                + feedback_pred
                + "\nEdit:"
            )

            if input_wfeed in self.cache:
                edit_pred = self.cache[input_wfeed]
            else:
                # Query GPT-3
                edit_pred = get_generations_gpt3(
                    ls=[input_wfeed],
                    model_name=self.model_name,
                    clean_tok=True,
                    stop=[self.separator],
                    temperature=0.0,
                    batch_size=20,
                    max_length=50,
                    penalty=0.5,
                    n=1,
                    keyfile=self.openai_api_key,
                )
                edit_pred = edit_pred[0]
                edit_pred = edit_pred.strip()  # Strip whitespace.
                self.cache[input_wfeed] = edit_pred

            # Reward
            edit_gold = current_observation.target_or_reference_texts[0]
            reward = self.metric([edit_pred], [[edit_gold]])
            # print("{}\n{}\n{}\n".format(prompt_or_input_text, feedback_pred, edit_gold))
            # print("{}\t{}".format(edit_pred, reward))
            return reward + 0.0

        return 0


# TODO: can we do batched?
# TODO num of environs = 10
# (can this be a list to track
# rouge on feedback_pred vs feedback_gold and edit_pred vs edit_gold?)
