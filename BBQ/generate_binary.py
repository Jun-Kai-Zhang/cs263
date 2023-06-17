import json
import os
from math import floor

src_dir = "./BBQ/BBQ/data"
lst_files = os.listdir("./BBQ/BBQ/data")
context_train = open("./input0/train.txt", 'w')
context_valid = open("./input0/valid.txt", 'w')
context_test = open("./input0/test.txt", 'w')
question_train = open("./input1/train.txt", 'w')
question_valid = open("./input1/valid.txt", 'w')
question_test = open("./input1/test.txt", 'w')
label_train = open("./label/train.txt", 'w')
label_valid = open("./label/valid.txt", 'w')
label_test = open("./label/test.txt", 'w')

for src_file in lst_files:
    f_src = open(src_dir + '/' + src_file)
    lines = f_src.readlines()
    num_entries = len(lines) // 4
    for i in range(4 * floor(0.8 * num_entries)):
        entry = json.loads(lines[i])
        context_train.write("%s\n" % entry["context"])
        question_train.write("%s 0: %s. 1: %s. 2: %s.\n" % (entry["question"],\
        entry["ans0"], entry["ans1"], entry["ans2"]))
        label_train.write("%d\n" % entry["label"])
    for i in range(4 * floor(0.8 * num_entries), 4 * floor(0.9 * num_entries)):
        entry = json.loads(lines[i])
        context_valid.write("%s\n" % entry["context"])
        question_valid.write("%s 0: %s. 1: %s. 2: %s.\n" % (entry["question"],\
        entry["ans0"], entry["ans1"], entry["ans2"]))
        label_valid.write("%d\n" % entry["label"])
    for i in range(4 * floor(0.9 * num_entries), 4 * num_entries):
        entry = json.loads(lines[i])
        context_test.write("%s\n" % entry["context"])
        question_test.write("%s 0: %s. 1: %s. 2: %s.\n" % (entry["question"],\
        entry["ans0"], entry["ans1"], entry["ans2"]))
        label_test.write("%d\n" % entry["label"])

context_train.close()
context_valid.close()
question_train.close()
question_valid.close()
label_train.close()
label_valid.close()
    
