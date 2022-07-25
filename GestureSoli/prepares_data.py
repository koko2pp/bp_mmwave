import numpy as np

root= "Data/Gestures_ZTE/"

who = "dzh_"

gesture_type_all = ["knock", "lswipe", "rswipe", "lcircle", "rcircle"]
gesture_type_swipe = ["lswipe", "rswipe"]
gesture_type_circle = ["lcircle", "rcircle"]

gesture_num = 500

## 3 classification - knock, swipe, circle
#three_data = []
#three_label = []
#
#for tp in gesture_type_all:
#    for i in range(gesture_num):
#        if tp == "knock":
#            three_label.append("0")
#        elif tp == "lswipe" or tp == "rswipe":
#            three_label.append("1")
#        elif tp == "lcircle" or tp == "rcircle":
#            three_label.append("2")
#        
#        if i < 10:
#            id_ = "_00" + str(i)
#        elif i >= 10 and i < 100:
#            id_ = "_0" + str(i)
#        else:
#            id_ = "_" + str(i)
#        file_name = root + who + tp + id_ + ".txt"
#        raw_data = np.loadtxt(file_name).reshape(240, -1).reshape(20, -1)
#        raw_data = np.delete(raw_data, -2, axis=1)
#        raw_data = raw_data.reshape(220)
#        raw_data = np.rint(raw_data)
#        raw_data = raw_data.tolist()
#        to_save = []
#        for i in raw_data:
#            to_save.append(str(int(i)))
#        three_data.append(' '.join(to_save))
#
#with open("Data/three_data_0610.txt", 'w', encoding='utf-8') as f:
#    f.write('\n'.join(three_data))
#
#with open("Data/three_label_0610.txt", 'w', encoding='utf-8') as f:
#    f.write('\n'.join(three_label))

## 2 classification - lswipe, rswipe
#two_data = []
#two_label = []
#
#for tp in gesture_type_swipe:
#    for i in range(gesture_num):
#        if tp == "lswipe":
#            two_label.append("0")
#        elif tp == "rswipe":
#            two_label.append("1")
#        
#        if i < 10:
#            id_ = "_00" + str(i)
#        elif i >= 10 and i < 100:
#            id_ = "_0" + str(i)
#        else:
#            id_ = "_" + str(i)
#        file_name = root + who + tp + id_ + ".txt"
#        raw_data = np.loadtxt(file_name).reshape(240, -1).reshape(20, -1)
#        raw_data = np.delete(raw_data, -2, axis=1)
#        raw_data = raw_data.reshape(220)
#        raw_data = np.rint(raw_data)
#        raw_data = raw_data.tolist()
#        to_save = []
#        for i in raw_data:
#            to_save.append(str(int(i)))
#        two_data.append(' '.join(to_save))
#
#with open("Data/swipe_data_0610.txt", 'w', encoding='utf-8') as f:
#    f.write('\n'.join(two_data))
#
#with open("Data/swipe_label_0610.txt", 'w', encoding='utf-8') as f:
#    f.write('\n'.join(two_label))

# 2 classification - lcircle, rcircle
two_data = []
two_label = []

for tp in gesture_type_circle:
    for i in range(gesture_num):
        if tp == "lcircle":
            two_label.append("0")
        elif tp == "rcircle":
            two_label.append("1")
        
        if i < 10:
            id_ = "_00" + str(i)
        elif i >= 10 and i < 100:
            id_ = "_0" + str(i)
        else:
            id_ = "_" + str(i)
        file_name = root + who + tp + id_ + ".txt"
        raw_data = np.loadtxt(file_name).reshape(240, -1).reshape(20, -1)
        raw_data = np.delete(raw_data, -2, axis=1)
        raw_data = raw_data.reshape(220)
        raw_data = np.rint(raw_data)
        raw_data = raw_data.tolist()
        to_save = []
        for i in raw_data:
            to_save.append(str(int(i)))
        two_data.append(' '.join(to_save))

with open("Data/circle_data_0610.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(two_data))

with open("Data/circle_label_0610.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(two_label))
