import json

# Load JSON data from a file
with open('/kuacc/users/dkukul19/hpc_run/LLaVA-34b/caption_contest_results[200-299].json', 'r') as file:
    data = json.load(file)

# Initialize a list to hold the contest numbers
contest_numbers = []

# Extract the contest_number from each entry and add to the list
for entry in data:
    contest_numbers.append(entry['contest_number'])

# Print the list of contest numbers
print(contest_numbers)
