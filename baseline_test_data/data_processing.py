# import json

# # Input and output file paths
# input_file = "optmath_bench.jsonl"
# output_file = "optmath_bench.json"

# # Read JSONL and convert to JSON
# with open(input_file, "r", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# # Write as a JSON array
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print(f"Converted {input_file} to {output_file}")

# convert optmath_bench keys question and answer to description and ground_truth
import json

input_file = "optmath_bench.jsonl"
output_file = "optmath_bench.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

for item in data:
    item["description"] = item["question"]
    item["ground_truth"] = item["answer"]

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Converted {input_file} to {output_file}")