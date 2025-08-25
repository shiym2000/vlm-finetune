import json
from tqdm import tqdm

model_path = "/hdd/shiym/work_dirs/vlm-finetune/trl/ctrate-qwenvl-32-364-rg"
input_path = f"{model_path}/eval/output.json"
output_path = f"{model_path}/eval/output_format.json"

# Read the JSONL file and convert it to a list of dictionaries
with open(input_path, "r") as f:
    json_data = json.load(f)

data = [{"outputs": [{"value": {"generated_reports": []}}]}]
for d in tqdm(json_data):
    response = d["messages"][-1]["content"].strip()
    findings = ""
    impression = ""

    # 拆分字段内容（根据关键词）
    if "Findings:" in response:
        findings_part = response.split("Findings:")[-1]
        if "Impression:" in findings_part:
            findings, rest = findings_part.split("Impression:", 1)
            impression = rest
        else:
            findings = findings_part
    else:
        findings = response

    # 清理两段文字
    report_text = f"{findings.strip()} {impression.strip()}".strip()

    data[0]["outputs"][0]["value"]["generated_reports"].append(
        {
            "input_image_name": d["videos"][0].split("/")[-1].replace(".npy", ".nii.gz"),
            "report": report_text,
        }
    )

with open(output_path, "w") as f:
    json.dump(data, f, indent=4)
