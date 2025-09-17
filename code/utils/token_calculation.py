import json

#MBPP
data_path = 'data/mbpp.jsonl'
nums = []
with open(data_path, "r") as input_file:
    data_lines = input_file.readlines()
    for line in data_lines:
        data = json.loads(line)
        testcases = data['test_list']
        test_num = 0
        for test in testcases:
            test_num += len(list(test))
        num = len(list(data['text'])) + test_num
        nums.append(num)

 
print(sum(nums)/len(nums))

# NL2F
data_path = 'data/nl2f_test.json'
nums = []
with open(data_path, "r") as input_file:
    data = json.load(input_file)
    for elements in data:
        question_num = 0
        for dic in elements['t5Formulas']:
            question = dic['Question']
            # print(question)
            question_num = len(list(question)) 
            nums.append(question_num)
        
print(len(nums))
print(sum(nums)/len(nums))
 
#NL2Bash
data_path = 'data/nl2b_test.jsonl'
nums = []
with open(data_path, "r") as input_file:
    lines  = input_file.readlines()
    for jsonline in lines:
        question_num = 0
        elements = json.loads(jsonline)
        text = elements["messages"][2]["content"] 
        question_num = len(list(text)) 
        nums.append(question_num)
    
print(len(nums))
print(sum(nums)/len(nums))

