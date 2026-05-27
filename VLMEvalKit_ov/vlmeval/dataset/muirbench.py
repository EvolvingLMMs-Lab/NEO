import ast
import os
import string
from collections import OrderedDict

import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

from ..smp.file import load
from ..smp.misc import get_cache_path, toliststr
from .image_mcq import ImageMCQDataset


class MUIRBench(ImageMCQDataset):
    TYPE = "MCQ"

    DATASET_URL = {
        # TSV file located at: resources/MUIRBench_EASI.tsv
        "MUIRBench_EASI": "",
    }
    DATASET_MD5 = {
        "MUIRBench_EASI": None,
    }

    def _task_category(self):
        return [
            "Image-Text Matching",
            "Diagram Understanding",
            "Difference Spotting",
            "Visual Retrieval",
            "Counting",
            "Attribute Similarity",
            "Scene Understanding",
            "Action Understanding",
            "Geographic Understanding",
            "Visual Grounding",
            "Cartoon Understanding",
            "Ordering",
        ]

    def prepare_tsv(self, url, file_md5=None):
        data = super().prepare_tsv(
            self.DATASET_URL[self.dataset_name], self.DATASET_MD5[self.dataset_name]
        )
        # Data processing reference: https://github.com/EvolvingLMMs-Lab/EASI
        dataset_path = ""

        # === Transfer rel path to abs path ===
        if "image_path" in data.columns:

            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r"\/")))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if (
                    isinstance(p, str)
                    and p.strip().startswith("[")
                    and p.strip().endswith("]")
                ):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data["image_path"] = data["image_path"].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_path"])
        else:
            tgt_path = self.dump_image(line)

        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        # question text
        question_text = f"Question: {question}"

        # options text
        options_prompt = "Choices: \n"
        for key, item in options.items():
            options_prompt += f"({key}) {item}\n"

        post_prompt = "Hint: Please provide the correct option letter, such as A, B, C, D, directly.\nAnswer:"

        prompt = "\n".join([question_text, options_prompt, post_prompt])
        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures.
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split("<image>")
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type="text", value=part))
            if i < len(images):
                segs.append(dict(type="image", value=images[i]))

        return [s for s in segs if s["value"]]

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col="task",
            order=self._task_category(),
            dataset_name=getattr(self, "dataset_name", "MUIRBench"),
        )
