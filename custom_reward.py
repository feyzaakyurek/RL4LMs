from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.metric import BaseMetric, RougeMetric
from typing import Dict, Any, List
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from myutil import get_generations_gpt3, ForkedPdb
from numpy import mean
import json
import os, re

CALLS = 0


def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


class EditMatchMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.downstream_metric_name = kwargs["downstream_metric_name"]
        self.downstream_metric = metric_map[self.downstream_metric_name]
        self.prompt = kwargs["prompt_path"]
        self.separator = kwargs["separator"]
        self.openai_api_key = kwargs["openai_key"]
        self.model_name = kwargs["gpt3_model_name"]
        self.cache_path = kwargs["cache_path"]


        assert self.downstream_metric_name in "rouge_combined"

        # Load prompt.
        with open(self.prompt, "r") as f:
            self.prompt = f.read()

        # Check key is valid.
        if self.model_name != "code-davinci-002":
            assert self.openai_api_key == "openai_key_ai2"

        # Load cache from cache_path.
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        # Strip off task prefix
        inputs = [prompt.lstrip("Critique: ") for prompt in prompt_texts]

        # Prepend prompt.
        input_wfeed = [
            (
                self.prompt
                + self.separator
                + input_text
                + "\nFeedback: "
                + feedback_pred
                + "\nEdit:"
            )
            for input_text, feedback_pred in zip(inputs, generated_texts)
        ]

        if self.cache_path != "":
            # Check if we have cached results.
            # Cache queries.
            cache_queries = [
                (input_text + "\nFeedback: " + feedback_pred + "\nEdit:")
                for input_text, feedback_pred in zip(inputs, generated_texts)
            ]

            cached_results = []
            uncached_inputs = []
            for i, input in enumerate(cache_queries):
                if input in self.cache:
                    cached_results.append((i, self.cache[input]))
                else:
                    uncached_inputs.append((i, input_wfeed[i]))
            input_wfeed = [x[1] for x in uncached_inputs]

        # Query GPT-3
        edit_pred = get_generations_gpt3(
            ls=input_wfeed,
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

        if self.cache_path != "":
            # Update cache.
            uncached_queries = [cache_queries[i] for i, _ in uncached_inputs]
            self.cache.update(dict(zip(uncached_queries, edit_pred)))

            global CALLS
            if CALLS % 500 == 0:
                with open(self.cache_path, "w") as f:
                    json.dump(self.cache, f)
            CALLS += 1

            edit_pred = iter(edit_pred)
            uncached_results = [(i, next(edit_pred)) for i, _ in uncached_inputs]

            # Combine cached and uncached results.
            results = cached_results + uncached_results

            # Sort results by index.
            results.sort(key=lambda x: x[0])
            edit_pred = [v for _, v in results]

        scores = self.downstream_metric(edit_pred, reference_texts)
        em_cm = [
            exact_match_scripting(pred, gold) for pred, gold in zip(edit_pred, reference_texts)
        ]
        em = mean([e[0] for e in em_cm])
        custom = mean([e[1] for e in em_cm])
        scores.update({"exact_match": em, "custom_step": custom})
        metric_dict = {}
        for k, score in scores.items():
            metric_dict.update({f"custom_metrics/editmatch_{k}": (None, score)})
        return metric_dict


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
                prompt_texts=[prompt_or_input_text],
                generated_texts=[feedback_pred],
                reference_texts=[edit_gold],
            )
            reward = metric_dict[
                f"custom_metrics/editmatch_{self.metric.downstream_metric_name}"
            ][-1]
            return reward

        return 0


def rouge1_metric(pred: List[str], ref: List[List[str]]):
    res = RougeMetric().compute(
        prompt_texts=[], generated_texts=pred, reference_texts=ref
    )
    return res["lexical/rouge_rouge1"][-1]


def rouge_combined(pred: List[str], ref: List[List[str]]):

    rouge_keys = ["rouge1", "rouge2", "rougeL"]
    res = RougeMetric().compute(
        prompt_texts=[], generated_texts=pred, reference_texts=ref
    )
    rouge_scores = [res["lexical/rouge_" + k][-1] for k in rouge_keys]
    scores = dict(zip(rouge_keys, rouge_scores))
    scores.update({"rouge_combined": mean(rouge_scores)})
    return scores


def custom_metric_scripting_func(pred: str, gold: str):
    """
    Args:
        pred: a string, should be in functional format e.g, [INSERT] node1 [AFTER] node2 [END]
        gold: a string, should be in natural language format e.g, Insert node1 after node2
    """
    score = 0.2
    pred = pred.replace("'", "")
    gold = gold.replace("'", "")

    try:
        if "[INSERT]" in pred and "[INSERT]" in gold:

            if "[AFTER]" in pred and "[AFTER]" in gold:

                node_insert = (
                    re.search("\[INSERT\](.*)\[AFTER\]", pred).group(1).strip()
                )
                node_after = re.search("\[AFTER\](.*)\[END\]", pred).group(1).strip()
                node_insert_g = (
                    re.search("\[INSERT\](.*)\[AFTER\]", gold).group(1).strip()
                )
                node_after_g = re.search("\[AFTER\](.*)\[END\]", gold).group(1).strip()

                if node_insert == node_insert_g:
                    score += 0.4
                if node_after == node_after_g:
                    score += 0.4

            elif "[BEFORE]" in pred and "[BEFORE]" in gold:

                node_insert = (
                    re.search("\[INSERT\](.*)\[BEFORE\]", pred).group(1).strip()
                )
                node_before = re.search("\[BEFORE\](.*)\[END\]", pred).group(1).strip()
                node_insert_g = (
                    re.search("\[INSERT\](.*)\[BEFORE\]", gold).group(1).strip()
                )
                node_before_g = (
                    re.search("\[BEFORE\](.*)\[END\]", gold).group(1).strip()
                )

                if node_insert == node_insert_g:
                    score += 0.4
                if node_before == node_before_g:
                    score += 0.4

            else:
                return score

        elif "[REMOVE]" in pred and "[REMOVE]" in gold:

            node_remove = re.search("\[REMOVE\](.*)\[END\]", pred).group(1).strip()
            node_remove_g = re.search("\[REMOVE](.*)\[END\]", gold).group(1).strip()

            if node_remove == node_remove_g:
                score += 0.8

        elif "[REORDER]" in pred and "[REORDER]" in gold:

            node_reorder1 = re.search("\[REORDER\](.*)\[AND\]", pred).group(1).strip()
            node_reorder2 = re.search("\[AND\](.*)\[END\]", pred).group(1).strip()

            node_reorder1_g = re.search("\[REORDER\](.*)\[AND\]", gold).group(1).strip()
            node_reorder2_g = re.search("\[AND\](.*)\[END\]", gold).group(1).strip()

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


def exact_match_scripting(pred: str, gold: List[str]):
    score_list = []
    for ref in gold:
        score_list.append(custom_metric_scripting_func(pred, ref))

    score = max(score_list)

    return (score == 1.0) * 1.0, score


metric_map = {
    "custom": custom_metric_scripting_func,
    "rouge1": rouge1_metric,
    "rouge_combined": rouge_combined,
}


# TODO: can we do batched?
# TODO num of environs = 10
# (can this be a list to track
# rouge on feedback_pred vs feedback_gold and edit_pred vs edit_gold?)
if __name__ == "__main__":
    metric = EditMatch()
    print("hello world")

    args = {
        "downstream_metric_name": "rouge_combined",
        "prompt_path": "data/interscript/prompts_edit_functional.txt",
        "separator": "\n\n---\n\n",
        "openai_key": "openai_key_me",
        "gpt3_model_name": "code-davinci-002",
        "cache_path": "data/interscript/cache.json",
        # "cache_path": "data/interscript/cache_prompts_edit_functional_test.json",
    }

    metric = EditMatchMetric(**args)
    metric_dict = metric.compute(
        prompt_texts=[
            "Critique: Goal: plug in nightlight Steps: 1. find pillows and blankets 2. walk to nightlight 3. push button light on",
            "Critique: Goal: bring baby home Steps: 1. take baby 2. drop baby",
        ],
        generated_texts=["Should plug in the light", "you should drive home"],
        reference_texts=[
            ["[REMOVE] nightlight [END]"],
            ["[INSERT] drive home [AFTER] take baby [END]"],
        ],
    )
    print(metric_dict)
    print(CALLS)
