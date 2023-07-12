import radiate_sdk.radiate as rd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import imageio
import matplotlib.pyplot as plt

def assign_class_num(class_name):
    if class_name == "car":
        return 0
    elif class_name == "van":
        return 1
    elif class_name == "truck":
        return 2
    elif class_name == "bus":
        return 3
    elif class_name == "motorbike":
        return 4
    elif class_name == "pedestrian":
        return 5
    elif class_name == "g_of_pedestrians":
        return 6
    else:
        print(class_name)


def show_frame(frame=0):
    while(1):
        lidar_frame = str(frames[frame][0])
        camera_frame = str(frames[frame][1])
        
        data = np.genfromtxt(folder+"velo_lidar/"+lidar_frame+".csv",delimiter=",")
        points = data[:,0:3]
        for point in points:
            if point[1]<0.4:
                point[2]=-10
                point[0]=-10

        image = cv2.imread(folder+"zed_right/"+camera_frame+".png")
        image_array = np.array(image)
        

        lidar_proj = seq.project_lidar(data,seq.calib.LidarToRight,seq.calib.right_cam_mat)
        fusion = seq.overlay_camera_lidar(image_array,lidar_proj)


        cv2.imshow("fusion",fusion)
        cv2.imshow("lidar",proj)
        k=cv2.waitKey(0)
        print(k)
        if k==49:
            frame+=10
        if k==50:
            frame-=10
            

        if k==-1:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#mode = same, pseudo_distance, distance
def export_image(lidar,camera,mode):

        data = np.genfromtxt(folder+"velo_lidar/"+lidar+".csv",delimiter=",")
        points = data[:,0:3]
        for point in points:
            if point[1]<0.4:
                point[2]=-10
                point[0]=-10

        image = cv2.imread(folder+"zed_right/"+camera+".png")
        image_array = np.array(image)
        img = image_array.astype(np.float32)/255
        
        lidar_proj = seq.project_lidar(data,seq.calib.LidarToRight,seq.calib.right_cam_mat,mode)
        if (mode =="distance"):
            dist = np.expand_dims(lidar_proj,axis=2)
            dist = dist.astype(np.float32) / 65535
            #cv2.imwrite("depth.png",dist)
            fusion = np.dstack((image_array,dist))
            #rgbd = cv2.cvtColor(fusion, cv2.COLOR_BGRA2RGBA)
            cv2.imwrite("fused_images_"+mode+"/"+lidar+"+"+camera+".png",fusion)

            #four_channel_image_rgb = cv2.cvtColor(rgbd, cv2.COLOR_RGBA2RGB).astype(int)
            #print(four_channel_image_rgb)
            #plt.imshow(four_channel_image_rgb)
            #plt.show()
        else:
            fusion = seq.overlay_camera_lidar(image_array,lidar_proj)
            if (mode == "same"):
                cv2.imwrite("fused_images/"+lidar+"+"+camera+".png",fusion)
            else:
                cv2.imwrite("fused_images_"+mode+"/"+lidar+"+"+camera+".png",fusion)
        print(lidar+"+"+camera+".png")
        return

def export_yolo_annotation(lidar,camera,timestamp):
    image = cv2.imread(folder+"zed_right/"+camera+".png")
    image_array = np.array(image)

    file = open("yolo_annotations/"+lidar+"+"+camera+".txt","w")
    
    info = seq.get_from_timestamp(float(timestamp))
    if info == {}:
        return
    
    bboxes = seq.project_bboxes_to_camera(info['annotations']['lidar_bev_image'],
                                      seq.calib.right_cam_mat,
                                      seq.calib.RadarToRight)
    

    #if len(bboxes)>0:
        #file2 = open("yolo_annotations/only_with_annotations/"+lidar+"+"+camera+".txt","w")
    
    for i in range(len(bboxes)):
        
        bbox = [bboxes[i]]
        
        bbox_array = bbox[0]['bbox_3d']
        if bbox_array.size==0:
            continue
        
        xmin = np.min(bbox_array[:,0])
        ymin = np.min(bbox_array[:,1])
        xmax = np.max(bbox_array[:,0])
        ymax = np.max(bbox_array[:,1])
        
        class_num = assign_class_num(bbox[0]['class_name'])
        
        width = abs(xmax-xmin)/image_array.shape[1]
        height = abs(ymax-ymin)/image_array.shape[0]
        center_x = ((xmax+xmin)/2)/image_array.shape[1]
        center_y = ((ymax+ymin)/2)/image_array.shape[0]
        
        yolo_bbox=[class_num, center_x,center_y,width,height]
        line = str(yolo_bbox[0])+" "+str(yolo_bbox[1])+" "+str(yolo_bbox[2])+" "+str(yolo_bbox[3])+" "+str(yolo_bbox[4])+"\n"
        file.write(line)
        #file2.write(line)
        
        new_bbox = np.array([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]])
        bbox[0]['bbox_3d'] = new_bbox
        #cv2.imshow("picture",seq.vis_3d_bbox_cam(image_array,bbox))

    file.close()
    #if  len(bboxes)>0:
        #file2.close()
    return

def copy_paste_images(frames):
    train,test = train_test_split(frames,random_state=45,test_size=0.2,shuffle=True)
    for frame in train:
        
        image = cv2.imread("fused_images/"+frame[0]+"+"+frame[1]+".png")
        cv2.imwrite("YOLO_Experiments/All_images/images/train/"+frame[0]+"+"+frame[1]+".png",image)

        file = open("yolo_annotations/"+frame[0]+"+"+frame[1]+".txt","r")
        lines = file.readlines()

        new_file = open("YOLO_Experiments/All_images/labels/train/"+frame[0]+"+"+frame[1]+".txt","w")
        for line in lines:
            new_file.write(line)

    for frame in test:
        
        image = cv2.imread("fused_images/"+frame[0]+"+"+frame[1]+".png")
        cv2.imwrite("YOLO_Experiments/All_images/images/test/"+frame[0]+"+"+frame[1]+".png",image)

        file = open("yolo_annotations/"+frame[0]+"+"+frame[1]+".txt","r")
        lines = file.readlines()

        new_file = open("YOLO_Experiments/All_images/labels/test/"+frame[0]+"+"+frame[1]+".txt","w")
        for line in lines:
            new_file.write(line)
        
    
    
    

folder = "city_1_0/"
seq = rd.Sequence(folder)

frames_txt = open("lidar_to_camera_frame.txt","r")

lines = frames_txt.readlines()
frames = []
for line in lines:
    temp = line.split()
    frames.append([temp[1],temp[3],temp[5]])
    
    #export_yolo_annotation(temp[1],temp[3],temp[5])
    
    export_image(temp[1],temp[3],"distance")
    
#show_frame()

#copy_paste_images(frames)

#A = export_image(temp[1],temp[3])
#cv2.imshow("image",A)


