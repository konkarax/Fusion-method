import radiate_sdk.radiate as rd
import numpy as np
import cv2


def project_2(seq,data,x,y,z,tx,ty,tz):

    

    rotation = np.array([[np.cos(z)*np.cos(y),
                          np.cos(z)*np.sin(y)*np.sin(x) - np.sin(z)*np.cos(x),
                          np.cos(z)*np.sin(y)*np.cos(x) + np.sin(z)*np.sin(x)],
                         [np.sin(z)*np.cos(y),
                          np.sin(z)*np.sin(y)*np.sin(x) + np.cos(z)*np.cos(x),
                          np.sin(z)*np.sin(y)*np.cos(x) - np.cos(z)*np.sin(x)],
                         [-np.sin(y),
                          np.cos(y)*np.sin(x),
                          np.cos(y)*np.cos(x)]])

    translation = np.array([tx, ty, tz])

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation

    fx = 3.379191448899105e+02
    fy = 3.386957068549526e+02
    cx = 3.417366010946575e+02
    cy = 2.007359735313929e+02

    
    fx = 337.873451599077
    fy = 338.530902554779
    cx = 329.137695760749
    cy = 186.166590759716

    intrinsic = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    #lidar_proj = seq.project_lidar(data,extrinsics,intrinsic,"same")
    lidar_proj = seq.project_lidar(data,seq.calib.LidarToRight,seq.calib.right_cam_mat)
    return lidar_proj

def show_frame(frame=0):
    x,y,z=1.57878, -0.00042,-0.0008
    tx,ty,tz =-0.06029, -0.14978723, 0.63788
    while(1):
        print(x,y,z)
        print(tx,ty,tz)
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
        
        proj = project_2(seq,data,x,y,z,tx,ty,tz)
        fusion = seq.overlay_camera_lidar(image_array,proj)


        cv2.imshow("fusion",fusion)
        cv2.imshow("lidar",proj)
        k=cv2.waitKey(0)
        print(k)
        if k==119:
            x+=0.01
        if k==115:
            x-=0.01
        if k==97:
            y+=0.01
        if k==100:
            y-=0.01
        if k==113:
            z+=0.01
        if k==101:
            z-=0.01


        if k==105:
            tx+=0.1
        if k==107:
            tx-=0.1
        if k==106:
            ty+=0.1
        if k==108:
            ty-=0.1
        if k==117:
            tz+=0.1
        if k==111:
            tz-=0.1

        if k==49:
            frame+=10
        if k==50:
            frame-=10
            
        if k==-1:
            break



    cv2.waitKey(0)
    cv2.destroyAllWindows()




folder = "city_1_0/"
seq = rd.Sequence(folder)

frames_txt = open("lidar_to_camera_frame.txt","r")

lines = frames_txt.readlines()
frames = []
for line in lines:
    temp = line.split()
    frames.append([temp[1],temp[3]])
show_frame(0)




