import torch

# Funtion to load classnames from a text document
def load_class_names(file):
    class_names = []
    with open(file,'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def boxes_iou(box1,box2):
    ''' Function to calculate the Intersection Of Union of two given
    bounding boxes'''

    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]

    area_box1 = width_box1*height_box1
    area_box2 = width_box2*height_box2

    mx = min(box1[0]-width_box1/2.0 , box2[0]-width_box2/2.0)
    Mx = max(box1[0]+width_box1/2.0,box2[0]+width_box2/2.0)

    union_width = Mx-mx

    my = min(box1[1]-height_box1/2.0 , box2[1]-height_box2/2.0)
    My = max(box1[1]+height_box1/2.0,box2[1]+height_box2/2.0)

    union_height = My-my

    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height

    if intersection_width<=0 or intersection_height<=0:
        return 0.0

    intersection_area = intersection_height*intersection_width

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area/union_area

    return iou


def nms(boxes,iou_thresh):
    '''Funtion to calculate the Non-Maximal suppression
       between two boxes'''
       
    if(len(boxes) == 0):
        return boxes

    det_cnfs = torch.zeros(len(boxes))

    for i in range(len(boxes)):
        det_cnfs[i] = boxes[i][4]

    _,sortIds = torch.sort(det_cnfs,descending=True)

    best_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]

        if box_i[4] > 0:
            best_boxes.append(box_i)
            for j in range(i+1,len(boxes)):
                box_j = boxes[sortIds[i]]
                if boxes_iou(box_i,box_j) > iou_thresh:
                    box_j[4] = 0
    return best_boxes