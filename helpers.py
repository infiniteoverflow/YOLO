import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Funtion to load classnames from a text document
def load_class_names(file):
    class_names = []
    with open(file,'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def boxes_iou(box1, box2):
  
    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx
    
    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Calculate the height of the union of the two bounding boxes
    union_height = My - my
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
   
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area/union_area
    
    return iou



def nms(boxes, iou_thresh):
    
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    
    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    _,sortIds = torch.sort(det_confs,descending=True)

    best_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]

        if box_i[4] > 0:
            best_boxes.append(box_i)
            for j in range(i+1,len(boxes)):
                box_j = boxes[sortIds[j]]
                if boxes_iou(box_i,box_j) > iou_thresh:
                    box_j[4] = 0
    return best_boxes


def detect_objects(model,img,iou_thresh,nms_thresh):
    start = time.time()

    model.eval()

    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)

    list_boxes = model(img,nms_thresh)

    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    boxes = nms(boxes,iou_thresh)

    finish = time.time()

    print("No. of Objects detected: {}".format(len(boxes)))
    print('Time Taken: {} sec'.format((finish-start)))

    return boxes

def print_objects(boxes, class_names):    
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))

            
def plot_boxes(img, boxes, class_names, plot_labels, color = None):
    
    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    
    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))
        
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        
        return int(r * 255)
    
    # Get the width and height of the image
    width = img.shape[1]
    height = img.shape[0]
    
    # Create a figure and plot the image
    fig, a = plt.subplots(1,1)
    a.imshow(img)
    
    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):
        
        # Get the ith bounding box
        box = boxes[i]
        
        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image. 
        x1 = int(np.around((box[0] - box[2]/2.0) * width))
        y1 = int(np.around((box[1] - box[3]/2.0) * height))
        x2 = int(np.around((box[0] + box[2]/2.0) * width))
        y2 = int(np.around((box[1] + box[3]/2.0) * height))
        
        # Set the default rgb value to red
        rgb = (1, 0, 0)
            
        # Use the same color to plot the bounding boxes of the same object class
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue  = get_color(0, offset, classes) / 255
            
            # If a color is given then set rgb to the given color instead
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color
        
        # Calculate the width and height of the bounding box relative to the size of the image.
        width_x = x2 - x1
        width_y = y1 - y2
        
        # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
        # lower-left corner of the bounding box relative to the size of the image.
        rect = patches.Rectangle((x1, y2),
                                 width_x, width_y,
                                 linewidth = 2,
                                 edgecolor = rgb,
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
        
        # If plot_labels = True then plot the corresponding label
        if plot_labels:
            
            # Create a string with the object class name and the corresponding object class probability
            conf_tx = class_names[cls_id] + ': {:.1f}'.format(cls_conf)
            
            # Define x and y offsets for the labels
            lxc = (img.shape[1] * 0.266) / 100
            lyc = (img.shape[0] * 1.180) / 100
            
            # Draw the labels on top of the image
            a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize = 24, color = 'k',
                   bbox = dict(facecolor = rgb, edgecolor = rgb, alpha = 0.8))        
        
    plt.show()