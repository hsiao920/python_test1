import os
import cv2
import random

from ultralytics import YOLO

'''
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
'''

#from tracker import Tracker

video_path = os.path.join('.','','people_-_6387 (720p).mp4')
video_out_path=os.path.join(',','','out.mp4')

cap=cv2.VideoCapture(video_path)

ret,frame=cap.read()

cap_out=cv2.VideoWriter(video_out_path,cv2.VideoWriter_fourcc(*'MP4V'),
                        cap.get(cv2.CAP_PROP_FPS),(frame.shape[1],frame.shape[0]))
                                                                                                 


model=YOLO("yolov8n.pt")

#tracker=Tracker()

colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(10)]

i=0

while ret:
    
    results=model(frame)

    for result in results:
        detections=[]
        for r in result.boxes.data.tolist():
            i+=1
            x1,y1,x2,y2,score,class_id=r
            #detections.append([int(x1),int(y1),int(x2),int(y2),score])
            
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(colors[int(i%len(colors))]),3)
    
        '''
        tracker.update(frame,detections)

        for track in tracker.tracks:
            bbox=track.bbox
            x1,y1,x2,y2=bbox
            track_id=track.track_id
 
 
        #cv2.imshow('frame',frame)
        #cv2.waitKey(1)
        '''  

        cap_out.write(frame)

        ret,frame=cap.read()



 
cap.release()
cap_out.release()
cv2.destroyAllWindows()