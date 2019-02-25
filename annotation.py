import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import time
import cv2
from matplotlib.widgets import RectangleSelector
#from generate_xml import write_xml
#global constants
img = None
tl_list = []
br_list = []
object_list = []

#constants
image_folder = 'capture_images' #folder that have images
savedir = 'annotations' #this is the folder that saves all annotation files
obj = 'head'
path = './cropped3'#'C:\\Users\Mythra\Desktop\darkflow-master\cropped_4'
image = None
def line_select_callback(clk,rls):
    print(clk.xdata, clk.ydata)
    print(rls.xdata, rls.ydata)
    global tl_list
    global br_list
    global object_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)
def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    if event.key == 'q':
        for i in range(len(object_list)) :
            # slice the image
            crop_img = image[ tl_list[i][1]:br_list[i][1], tl_list[i][0]:br_list[i][0], :]
            #imwrite that image
            cv2.imshow('crop img :', crop_img)
            filename = os.path.join(path, time.strftime("%a%d%H_%M_%S")) +'.jpg'
            cv2.imwrite(filename,crop_img)
            time.sleep(1.5)
        #write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        tl_list = []
        br_list = []
        object_list = []
        img = None


def toggle_selector(event):
    toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1)
        if 'png' in image_file.path:
            image = cv2.imread(image_file.path)
            print('file path :',image_file.path)
            
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            
            toggle_selector.RS = RectangleSelector(
                ax, line_select_callback, drawtype = 'box', useblit= True,
                button=[1], minspanx=5, minspany = 5,
                spancoords = 'pixels', interactive = True
            )
            bbox = plt.connect('key_press_event',toggle_selector)
            key = plt.connect('key_press_event', onkeypress)
            plt.show()
            plt.close(fig)
