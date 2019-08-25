import json
import requests
import urllib.request
from lxml import etree as ET
from lxml.etree import Element, SubElement
from xml.dom import minidom
from tqdm import tqdm
from utils.bbox import BoundBox

loaded_json = open("Indian_Number_plates.json",'r')

loaded_json_file = loaded_json.readlines()

loaded_json.close()

def write_annotations(image_path, boxes, labels, obj_thresh,image_height,image_width):

    img_dict = {}
    name = image_path.split('/')[-1]
    for box in boxes:

        for i in range(len(labels)):   
            if box.classes[i] > obj_thresh:
                        x1 = box.xmin
                        y1 = box.ymin
                        x2 = box.xmax
                        y2 = box.ymax
                        klass = labels[i]
                        if not name in img_dict:
                            img_dict[name] = []
                        img_dict[name].append([klass, x1, y1, x2, y2])
    image_path_1 = image_path[:(len(image_path)-len(name))]
    folder_name = image_path_1.split('/')[-2]
    image_path_1 = image_path_1[:(len(image_path_1)-len(folder_name)-1)] +"train_ann/"
    parser = ET.XMLParser(remove_blank_text=True)
    for key in tqdm(img_dict):
        tree = ET.parse('ann_dummy.xml', parser)
        root = tree.getroot()
        root[1].text = key
        root[2].text = key
        num_objects = len(img_dict[key])
        dim_child = SubElement(root,'size')
        baby_dim_w = SubElement(dim_child,'width')
        baby_dim_w.text = str(image_width)
        baby_dim_h = SubElement(dim_child,'height')
        baby_dim_h.text = str(image_height)
        baby_dim_d = SubElement(dim_child,'depth')
        baby_dim_d.text = '3'
        for objecty in img_dict[key]:
            child = SubElement(root, 'object')
            baby1 = SubElement(child, 'name')
            baby1.text = objecty[0]
            baby2 = SubElement(child, 'pose')
            baby2.text = 'Unspecified'
            baby3 = SubElement(child, 'truncated')
            baby3.text = '0'
            baby4 = SubElement(child, 'difficult')
            baby4.text = '0'
            baby5 = SubElement(child, 'bndbox')
            baby6 = SubElement(baby5, 'xmin')
            baby6.text = str(objecty[1])
            baby7 = SubElement(baby5, 'ymin')
            baby7.text = str(objecty[2])
            baby8 = SubElement(baby5, 'xmax')
            baby8.text = str(objecty[3])
            baby9 = SubElement(baby5, 'ymax')
            baby9.text = str(objecty[4])
        tree.write('{}{}.xml'.format(image_path_1,key[:-5]), pretty_print=True)


i = 0
for x in loaded_json_file:
    y = json.loads(x)
    i+=1
    img_loc_url = y["content"]
    save_loc = str("C:/Games/Projects/TCS_Project/train_imgs/"+str(i)+".jpeg")
    urllib.request.urlretrieve(img_loc_url, save_loc)
    image_height = y["annotation"][0]["imageHeight"]
    image_width = y["annotation"][0]["imageWidth"]
    xmin,ymin = y["annotation"][0]["points"][0]["x"],y["annotation"][0]["points"][0]["y"]
    xmax,ymax = y["annotation"][0]["points"][1]["x"],y["annotation"][0]["points"][1]["y"]
    labels = y["annotation"][0]["label"]
    box = [BoundBox(int(xmin*image_width), int(ymin*image_height), int(xmax*image_width), int(ymax*image_height),None,[1])]
    write_annotations(save_loc,box,labels,0.5,int(image_height),int(image_width))




