import argparse
import json
from tqdm import tqdm


def eval(args):
    with open(args.output_path, "r") as f:
        output_list = json.load(f)

    expected_output_dict = {}
    for output in tqdm(output_list):
        expected_output_dict[output["id"]] = {
            "question": output["messages"][0]["content"].strip(),
            "answer": output["answer"].strip(),
            "predict": output["messages"][-1]["content"].strip(),
        }

    with open(args.reference_path, "r") as f:
        question_ids = json.load(f)
    question_ids = question_ids["test"]

    correct_list = []
    incorrect_list = []

    for question_id in tqdm(question_ids):
        question = expected_output_dict[question_id]["question"]
        answer = expected_output_dict[question_id]["answer"]
        predict = expected_output_dict[question_id]["predict"]
        is_multimodal = "<image>" in question

        result = {
            "id": question_id,
            "is_multimodal": is_multimodal,
            "predict": predict,
        }

        if f"{answer}." in predict or f" {answer} " in (" " + predict + " "):
            correct_list.append(result)
        else:
            incorrect_list.append(result)

    correct = len(correct_list)
    total = len(correct_list) + len(incorrect_list)
    mm_correct = len([x for x in correct_list if x["is_multimodal"]])
    mm_incorrect = len([x for x in incorrect_list if x["is_multimodal"]])
    mm_total = mm_correct + mm_incorrect
    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {mm_correct / mm_total * 100:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--reference_path", type=str, default=None)
    args = parser.parse_args()

    eval(args)
