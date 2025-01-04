# AI-Virtual-Keyboard

---
mediapipe URL : <https://google.github.io/mediapipe/>

---
Packages
```python
import cv2
import mediapipe as mp
import time
from IPython.display import clear_output
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
```

---
Testing webcam
```python
cap = cv2.VideoCapture('http://192.168.0.8:8080/video')
while True:
    success, image = cap.read()
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
```
I used "IP WebCam" in Google market.

---
Testing mediapipe
```python
cap = cv2.VideoCapture('http://192.168.0.8:8080/video')

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)

                clear_output(wait=True)
                print(hand_landmarks.landmark[8])
        
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()   
cap.release()
```

---
Thumb and other finger contact recognition
```python
def img_process(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def to_numpy(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])

def get_std_len(hand_landmarks):
    mark1 = to_numpy(hand_landmarks.landmark[0])
    mark2 = to_numpy(hand_landmarks.landmark[5])
    std_len = np.linalg.norm(mark1-mark2)
    return std_len


flags = [True] * 4

cap = cv2.VideoCapture('http://192.168.0.8:8080/video')
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
            
        image, results = img_process(image)

        if results.multi_hand_landmarks:
            clear_output(wait=True)
            print(results.multi_handedness[0].classification[0].label)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
                
                
                marks = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]
                locs = np.stack([to_numpy(hand_landmarks.landmark[i]) for i in marks])
                std_len = np.linalg.norm(locs[0] - locs[1])
                
                dists = np.linalg.norm(locs[5] - locs[6:], axis=1) / std_len
                
                for i in range(4):
                    print(dists[i])
                    if flags[i] and dists[i] < 0.35:
                        flags[i] = False
                    elif not flags[i] and 0.5 < dists[i]:
                        flags[i] = True
                        
                for i in range(4):
                    if flags[i]:
                        print('Realse %d'%i)
                    else :
                        print('Tap %d'%i)
                
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()   
cap.release()
```

---
Final code
```python
WIDTH, HEIGHT = 1280, 720

def xyxy_to_xywh(xyxy):
    center = (xyxy[2:] + xyxy[:2])//2
    wh = xyxy[2:] - xyxy[:2]
    return np.concatenate([center, wh], axis=-1)

def xywh_to_xyxy(xywh):
    xy1 = xywh[:2] - xywh[2:]//2
    xy2 = xywh[:2] + xywh[2:]//2
    return np.concatenate([xy1, xy2], axis=-1)

def draw_text(img, text, box_xywh):
    text_size = np.array(cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0])
    text_size[1] *= -1
    text_xy1 = box_xywh[:2] - text_size//2
    cv2.putText(img, text, text_xy1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, )
    
class BoxObject():
    def __init__(self, xywh, color=(0, 0, 255), text='', editable=False):
        self.xywh = np.array(xywh)
        self.color = color
        self.text = text
        self.editable = editable
        
    def draw(self, img):
        xyxy = xywh_to_xyxy(self.xywh)
        cv2.rectangle(img, xyxy[:2], xyxy[2:], self.color, -1)
        draw_text(img, self.text, self.xywh)
        
    def __repr__(self):
        return str((self.text, 'edit:%s'%self.editable))
    
class HandManager():
    def __init__(self, label):
        self.label = label
        self.locs = [None]*21
        self.std_len = None
        self.dists = None
        self.state = [0]*4 # 0-release, 1-click, 2-drag
        
    def update(self, marks):
        locs = np.stack([to_numpy(mark) for mark in marks])
        locs = np.clip(locs, 0, 1)
        locs[:,0] *= WIDTH-1
        locs[:,1] *= HEIGHT-1
        self.locs = locs.astype(np.int32)
        self.std_len = np.linalg.norm(locs[0] - locs[5])
        self.dists = np.linalg.norm(locs[4] - locs[[8,12,16,20]], axis=1) / self.std_len

        for i in range(4):
            if not self.state[i] and self.dists[i] < 0.35: self.state[i] = 1
            elif self.state[i] and 0.45 < self.dists[i]: self.state[i] = 0
            elif self.state[i] == 1: self.state[i] = 2
        
    def __repr__(self):
        return '#####\nlabel : {}\n검지위치 : {}\ndists : {}\nstate : {}'.format(self.label, self.locs[8], self.dists, self.state)
        
        

def img_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    return results

def to_numpy(landmark):
    return np.array([landmark.x, landmark.y])

def action(locs, board, hand_mng):
    global graped_box, selected_box
    
    x, y = hand_mng.locs[8]
    if hand_mng.state[1] == 1: # 클릭했을때
        if board[y, x]:
            box = box_obj[board[y, x]-1]
            if box.editable:
                selected_box = box
            graped_box = box
        else:
            selected_box = None
    elif hand_mng.state[1] == 2: # 드래그중일때
        if graped_box and graped_box.editable:
            graped_box.xywh[:2] = [x,y]
    else:
        graped_box = None

key_locs = [(900,100), (1010,100), (1120,100), 
           (900,210), (1010,210), (1120,210), 
           (900,320), (1010,320), (1120,320), 
           (900,430), (1010,430), (1120,430), ]
key_text = ['1','2','3','4','5','6','7','8','9','*','0','#',]
key_boxes = [BoxObject((x,y,100,100), text=n, editable=False) for (x,y),n in zip(key_locs,key_text)]

temp_box1 = BoxObject((110,110,200,200), text='box1', editable=True) # xywh
temp_box2 = BoxObject((330,110,200,200), text='', editable=True) # xywh
box_obj = [temp_box1, temp_box2]
box_obj.extend(key_boxes)

selected_box = None
graped_box = None

right_hand = HandManager('Right')
left_hand = HandManager('Left')
manager_dict = {'Right':right_hand, 'Left':left_hand}

cap = cv2.VideoCapture('http://192.168.0.8:8080/video')
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
            
        image = cv2.flip(image, 1)
        results = img_process(image)
        tmp = image.copy()
        
        board = np.zeros((HEIGHT,WIDTH), dtype=np.int8)
        for i, box in enumerate(box_obj, start=1):
            box.draw(tmp)
            xyxy = xywh_to_xyxy(box.xywh)
            x1,y1,x2,y2 = np.clip(xyxy, 0, WIDTH)
            board[y1:y2,x1:x2] = i
        
        cv2.addWeighted(tmp, 0.5, image, 0.5, 0, image)
            
        clear_output(wait=True)
        
        if results.multi_hand_landmarks:
            left_right = list(map(lambda x:x.classification[0].label, results.multi_handedness))
            if len(left_right)==1 and left_right[0]=='Right':
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
                right_hand.update(hand_landmarks.landmark)
        
                action(right_hand.locs, board, right_hand)
                if selected_box and graped_box and not graped_box.editable and right_hand.state[1]==1:
                    selected_box.text += graped_box.text
            
                print('click mode')
                    
            else:
                print('None mode')
                
            print(right_hand)
            
            print('graped_box:', graped_box)
            print('selected_box:', selected_box)
                    
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()   
cap.release()
```

---


![ai-virtual-keyboard](https://github.com/user-attachments/assets/4626f37c-8d29-4be5-bda8-5c60d3a8f2b6)




















