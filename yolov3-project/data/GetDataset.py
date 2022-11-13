
import glob
import random
import os
if __name__ == "__main__":
    txt_all=glob.glob('./*/*.txt')
    random.shuffle(txt_all)
    frac = int(0.8 * len(txt_all))
    f1 = open("subtitle_train.txt", 'w', encoding='utf-8')
    f2 = open("subtitle_val.txt", 'w', encoding='utf-8')
    for i, line in enumerate(txt_all):
        new_line = line.replace('labels','images').replace('txt','jpg')
        if i < frac:
            f1.write(os.path.abspath(new_line))
            f1.write('\n')
        else:
            f2.write(os.path.abspath(new_line))
            f2.write('\n')
    f1.close()
    f2.close()