import cv2
import os
import random

if __name__ == "__main__":
    sub_p = './subtitle'
    img_p = './nonsubtitle'
    des_p = './new_images'
    des_t = './new_labels'
    if not os.path.exists(des_p): os.mkdir(des_p)
    if not os.path.exists(des_t): os.mkdir(des_t)
    for img_ in os.listdir(img_p):
        img_path  =  os.path.join(img_p,img_)
        count = 0
        img = cv2.imread(img_path)
        h,w,c = img.shape
        for ti in range(0,100):
            sub_all = os.listdir(sub_p)
            ind = random.randint(0,len(sub_all)-1)
            sub = sub_all[ind]
            new_img = img.copy()
            sub_path = os.path.join(sub_p,sub)
            sub_img = cv2.imread(sub_path)
            sh,sw,sc = sub_img.shape
            lx = int((w - sw)/2)
            ly = int(h-2*sh)
            new_img[ly:ly+sh,lx:lx+sw] = sub_img
            name = img_.split('.jpg')[0]
            name  = name+'_'+ str(count)
            new_img_name = os.path.join(des_p,name+'.jpg')
            cv2.imwrite(new_img_name,new_img)
            txt_name = os.path.join(des_t,name+'.txt')
            nx = (lx+sw/2)/w
            ny = (ly+sh/2)/h
            nw = sw/w
            nh = sh/h
            f = open(txt_name,'w',encoding='utf-8')
            f.write('0 '+str(nx)+' '+str(ny)+' '+str(nw)+' '+str(nh))
            f.close()
            count += 1


