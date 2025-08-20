import argparse
import json
from tqdm import tqdm


def eval(args):
    with open(args.output_path, "r") as f:
        output_list = json.load(f)

    expected_output_list = []
    for output in tqdm(output_list):
        questionId = output["id"]
        prediction = output["messages"][-1]["content"].strip().lower()
        expected_output_list.append(dict(questionId=questionId, prediction=prediction))

    with open(args.expected_output_path, "w") as f:
        json.dump(expected_output_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--expected_output_path", type=str, default=None)
    args = parser.parse_args()

    eval(args)
