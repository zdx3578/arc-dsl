import json
import os
import sys

def convert_to_prolog(data):
    prolog_facts = []

    for example in data['train']:
        input_grid = example['input']
        output_grid = example['output']
        input_str = convert_grid_to_prolog(input_grid)
        output_str = convert_grid_to_prolog(output_grid)
        prolog_facts.append(f"pos(program({input_str}, {output_str})).")

    for example in data['test']:
        input_grid = example['input']
        output_grid = example['output']
        input_str = convert_grid_to_prolog(input_grid)
        output_str = convert_grid_to_prolog(output_grid)
        prolog_facts.append(f"test(program({input_str}, {output_str})).")

    return "\n".join(prolog_facts)

def convert_grid_to_prolog(grid):
    return "[" + ", ".join("[" + ", ".join(map(str, row)) + "]" for row in grid) + "]"

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_path = os.path.join(directory, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
            prolog_facts = convert_to_prolog(data)
            exs_filename = os.path.splitext(filename)[0] + '.pl'
            exs_path = os.path.join(directory, exs_filename)
            with open(exs_path, 'w') as f:
                f.write(prolog_facts)
            print(f"Prolog facts have been written to {exs_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_prolog.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    process_directory(directory)