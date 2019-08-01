import csv
# name = []
# file = open(r'D:\wrong_name.txt', 'r')
# for line in file.readlines():
#     name.append(line[:-1])
# file.close()
# file1 = open(r'D:\visual.txt', 'x')
# for i in range(len(name)):
#     for j in range(len(name[i])):
#         if name[i][j] == '_':
#             id = name[i][:j]
#             break
#     file1.write('/data/zhangshihao/AS-OCT-MK/data/AllImg-Resize/' + id + '/' + name[i] + '\n')
# file1.close()

susp_file = []
left = 85
right = 85
csv_file = csv.reader(open(r'C:\Users\cvter\Desktop\AS-OCTmk\SS detection\test\Total_data.csv', 'r'))
i = 0
for line in csv_file:
    if i == 0:
        i += 1
        continue
    pic_name = line[1]
    left_loss = float(line[3])
    right_loss = float(line[4])
    if left_loss < 0 or right_loss < 0:
        if pic_name not in susp_file:
            susp_file.append(pic_name)
    if left_loss > left or right_loss > right:
        if pic_name not in susp_file:
            susp_file.append(pic_name)
    i += 1

txt = open(r'D:\user5-8wrong_name.txt', 'x')
for name in susp_file:
    for j in range(len(name)):
        if name[j] == '_':
            id = name[:j]
            break
    txt.write('/data/zhangshihao/AS-OCT-MK/data/AllImg-Resize/' + id + '/' + name + '\n')
txt.close()
