import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from tqdm import tqdm


def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = osp.join(data_path, category)
        if not osp.isdir(category_dir):
            continue
        if osp.exists(osp.join(category_dir, "images")):
            image_path = osp.join(category_dir, "images")
            qa_path = osp.join(category_dir, "questions_answers_YN")
        else:
            image_path = qa_path = category_dir
        assert osp.isdir(image_path), image_path
        assert osp.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith(".txt"):
                continue
            for line in open(osp.join(qa_path, file)):
                question, answer = line.strip().split("\t")
                GT[(category, file, question)] = answer
    return GT


def eval(args):
    with open(args.output_path, "r") as f:
        output_list = json.load(f)

    expected_output_dict = defaultdict(list)
    for output in tqdm(output_list):
        category = output["id"].split("/")[0]
        file = output["id"].split("/")[-1].split(".")[0] + ".txt"
        question = output["messages"][0]["content"][8:]
        expected_output_dict[category].append((file, question, output["messages"][-1]["content"].strip()))

    GT = get_gt(args.reference_dir)

    os.makedirs(args.expected_output_dir, exist_ok=True)
    for category, cate_tups in expected_output_dict.items():
        with open(osp.join(args.expected_output_dir, f"{category}.txt"), "w") as fp:
            for file, prompt, answer in cate_tups:
                if "Answer the question using a single word or phrase." in prompt:
                    prompt = prompt.replace("Answer the question using a single word or phrase.", "").strip()
                if "Please answer yes or no." not in prompt:
                    prompt = prompt + " Please answer yes or no."
                    if (category, file, prompt) not in GT:
                        prompt = prompt.replace(" Please answer yes or no.", "  Please answer yes or no.")
                gt_ans = GT[category, file, prompt]
                tup = file, prompt, gt_ans, answer
                fp.write("\t".join(tup) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--expected_output_dir", type=str, default=None)
    parser.add_argument("--reference_dir", type=str, default=None)
    args = parser.parse_args()

    eval(args)
