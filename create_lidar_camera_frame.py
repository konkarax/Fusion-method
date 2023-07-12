def create_frames(lines):
    frames=[]
    for line in lines:
        temp = line.split()
        if len(temp[3])==19:
            num = temp[3].split(".")
            num[1] = "0" + num[1]
            new_num = num[0]+"."+num[1]
            temp[3] = new_num
        frames.append([int(temp[1]),temp[3]])
    return frames

def add_zeros(frame):
    frame=str(frame)
    add_zero = 6- len(frame)
    for i in range(add_zero):
        frame = "0"+frame
    return frame
    
    

def correct_file(file,frames):
    for frame in frames:
        frame[0] = add_zeros(frame[0])
        file.write("Frame: "+frame[0]+" Time: "+frame[1]+"\n")
    return
    

path = "city_1_0/"

pic_txt = open(path+"zed_right.txt","r")
lines = pic_txt.readlines()
pic_txt.close()
pic_frames=create_frames(lines)

pic_txt = open(path+"zed_right.txt","w")
correct_file(pic_txt,pic_frames)
pic_txt.close()
    
lidar_txt = open(path+"velo_lidar.txt","r")
lines = lidar_txt.readlines()
lidar_txt.close()
lidar_frames = create_frames(lines)

lidar_txt = open(path+"velo_lidar.txt","w")
correct_file(lidar_txt,lidar_frames)
lidar_txt.close()

i=0
j=0
lidar_camera=[]
for frame in lidar_frames:
    j+=1
    if float(frame[1])<float(pic_frames[i][1])-0.1:
        continue
    while(True):
        if float(frame[1])>float(pic_frames[i][1])+0.1:
            #print(frame[1],pic_frames[i][1])
            i+=1
            continue
        break
    lidar_camera.append([frame[0],pic_frames[i][0],pic_frames[i][1]])
    i+=1


file = open("lidar_to_camera_frame.txt","w")
for frames in lidar_camera:
    lidar_frame = add_zeros(frames[0])
    camera_frame = add_zeros(frames[1])
    timestamp = str(frames[2])
    file.write("Lidar_Frame: "+lidar_frame+" Camera_Frame: "+camera_frame+" Timestamp: "+timestamp+"\n")
    
file.close()
    
