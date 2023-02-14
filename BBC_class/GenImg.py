import os

f = open('data_list.txt','w',encoding='utf-8')
dst = './imgs/source'
files = os.listdir(dst)
for line in files:
    path = os.path.join(dst,line)
    path2 = path.replace('source','target')
    string = path +' ' + path2 
    if os.path.isfile(path) and os.path.isfile(path2):
        f.write(string)
    f.write('\n')
f.close()