from rl4lms.envs.text_generation.metric import BaseMetric, RougeMetric
from typing import List, Dict, Any
from transformers import PreTrainedModel
from myutil import get_generations_gpt3, ForkedPdb
from numpy import mean
import json
import os, re
import ipdb

CALLS = 0


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
        metric_dict = {}
        for k, score in scores.items():
            metric_dict.update({f"custom_metrics/editmatch_{k}": (None, score)})
        return metric_dict


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

if __name__ == "__main__":
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
