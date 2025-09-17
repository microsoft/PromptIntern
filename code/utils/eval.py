import json 

output_path = '...'


with open(output_path, "r") as input_file:
    correct_num = 0
    total_num = 0
    for i,line in enumerate(input_file):
        data = json.loads(line)
        if data["exit_code"] == 0: correct_num += 1
        total_num += 1
acc = correct_num / total_num
print(correct_num, total_num, acc)
       