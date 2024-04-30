import json

def load_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_file(data, file_path):

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

file1 = 'caption_contest_results[0-199].json'
file2 = 'caption_contest_results[200-299].json'
file3 = 'caption_contest_results[300-end].json'

data1 = load_json_file(file1)
data2 = load_json_file(file2)
data3 = load_json_file(file3)

combined_data = data1 + data2 + data3

combined_file_path = 'caption_contest_results_all.json'

save_json_file(combined_data, combined_file_path)

print("The JSON files have been successfully combined.")
