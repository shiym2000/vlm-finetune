import argparse
import json
from tqdm import tqdm


def eval(args):
    with open(args.output_path, "r") as f:
        output_list = json.load(f)

    expected_output_dict = {}
    for idx, output in tqdm(enumerate(output_list)):
        expected_output_dict[f"v1_{idx}"] = output["messages"][-1]["content"].strip()

    with open(args.expected_output_path, "w") as f:
        json.dump(expected_output_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--expected_output_path", type=str, default=None)
    args = parser.parse_args()

    eval(args)
