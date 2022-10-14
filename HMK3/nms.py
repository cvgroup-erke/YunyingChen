import numpy as np

def nms(dets,thresh=0.5, force_cpu=False):
    '''
    :param dets: [x1,y1,x2,y2,conf]
    :param thresh:  the threshold of iou
    :param force_cpu: using cpu or not
    :return: the index of bboxes which meet the requirement
    '''
    ## seperate the cordinate
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    conf = dets[:, 4]

    # calculate all the bbox area
    area = (y2-y1+1) * (x2- x1+1)
    # arrange the bbox with the largest confidence to the least
    index = conf.argsort()[::-1]
    final = []

    while len(index) > 0:
        i = index[0]
        final.append(i)
        # calculate the iou between the current bbox and the rest bbox
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22-x11+1)
        h = np.maximum(0, y22-y11+1)
        inter = w*h
        union = area[i] + area[index[1:]] - inter
        iou = inter/union
        idx = np.where(iou <= thresh)[0]
        index = index[idx+1]

    return final

def soft_nms(dets,sigma=0.5, threshold1=0.7, threshold2=0.1, method='linear'):
        '''
        :param dets: [x1,y1,x2,y2,conf]
        :param thresh:  the threshold of iou
        :param force_cpu: using cpu or not
        :return: the index of bboxes which meet the requirement
        '''
        ## seperate the cordinate
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        conf = dets[:, 4]

        # calculate all the bbox area
        area = (y2 - y1 + 1) * (x2 - x1 + 1)

        index = [i for i in range(0,dets.shape[0])]

        new_conf = list(conf.copy())
        final = [ ]

        while len(index) > 0:
            max_score = 0
            max_pos = -1

            for i in index:
                if new_conf[i] >= max_score:
                    max_pos = i
                    max_score = new_conf[i]

            if max_pos == -1:
                break

            final.append(max_pos)
            index.remove(max_pos)

            # calculate the iou between the max conf bbox and the rest bbox
            x11 = np.maximum(x1[max_pos], x1[index])
            y11 = np.maximum(y1[max_pos], y1[index])
            x22 = np.minimum(x2[max_pos], x2[index])
            y22 = np.minimum(y2[max_pos], y2[index])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            inter = w * h
            union = area[max_pos] + area[index] - inter

            ious = inter / union

            new_index = []
            for i, idx in enumerate(index):
                iou = ious[i]
                weight =1

                if method == 'gaussian':
                    weight = np.exp(-(iou*iou)/sigma)
                if method =='linear':
                    if iou >=threshold1:
                        weight = 1 - iou
                else:
                    print('Method Error')
                    return []

                new_conf[idx] = new_conf[idx] * weight

                if new_conf[idx] > threshold2:
                    new_index.append(idx)
            index = new_index
        return final

