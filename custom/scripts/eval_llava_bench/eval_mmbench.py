import argparse
import json
from tqdm import tqdm
import pandas as pd


def eval(args):
    with open(args.output_path, "r") as f:
        output_list = json.load(f)

    df = pd.read_table(args.reference_path)
    cur_df = df.copy()
    cur_df = cur_df.drop(columns=["hint", "category", "source", "image", "comment", "l2-category"])
    cur_df.insert(6, "prediction", None)

    count = 0
    for i, output in tqdm(enumerate(output_list)):
        cur_df.loc[i, "prediction"] = output["messages"][-1]["content"].strip()
        # print(cur_df.loc[i, 'answer'])
        # print(pred['outputs'][1:2])

        if output["messages"][-1]["content"].strip() == cur_df.loc[i, "answer"].strip():
            count += 1

    print(len(output_list), count / len(output_list))

    cur_df.to_excel(args.expected_output_path, index=False, engine="openpyxl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--expected_output_path", type=str, default=None)
    parser.add_argument("--reference_path", type=str, default=None)
    args = parser.parse_args()

    eval(args)
