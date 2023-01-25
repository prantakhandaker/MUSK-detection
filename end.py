from keras.models import load_model
import cv2
import numpy as np


model = load_model('model-020.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap =cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


while(True):

    success,img=cap.read()
    img = cv2.resize(img,(800,700)) # (400,800), (800,700)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1)) #reshape to 4D
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0] #to determine which has the maximum probability
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2) #for bounding box
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1) #for closed or filled rectangle on top of bounding box
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # text to be displayed
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
# source.release()