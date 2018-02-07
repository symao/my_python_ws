import cv2

def video_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. exit.")
        exit()
    writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*"XVID"), 30, (640,480))
    ret,img = cap.read()
    while ret:
        writer.write(img)
        cv2.imshow('img',img)
        key = cv2.waitKey(30)
        if key == 27:
            break
        ret,img = cap.read()

if __name__ == '__main__':
    video_capture()
    