# merge_json_lists.py
# Usage: python merge_json_lists.py out.json in1.json in2.json ...
import os 
import json, sys
import argparse

# add argument parsing
parser = argparse.ArgumentParser(description="Merge multiple JSON list files into one.")
parser.add_argument("--directory", help="Directory containing input JSON files")
parser.add_argument("--output", help="Output JSON file path")
parser.add_argument("--inputs", nargs="+", help="Input JSON file paths")
args = parser.parse_args()

if len(args.inputs) < 1:
    print("Usage: python merge_json_lists.py out.json in1.json [in2.json ...]")
    raise SystemExit(1)

directory = args.directory
out_path = args.output
all_items = []

for p in args.inputs:
    with open(os.path.join(directory,p), "r+") as f:
        print('loading', os.path.join(directory,p))
        # load a list of json objects, each line is a json object
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            all_items.append(data)

with open(out_path, "w", encoding="utf-8") as f:
    # write each item as a json object in a new line
    f.writelines([json.dumps(item) + "\n" for item in all_items])

print(f"Merged {len(args.inputs)} files -> {out_path} with {len(all_items)} items")
