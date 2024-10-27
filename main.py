import os
import cv2
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            #decimal values

            #they have to be converted to integer pixels
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30,30))
    
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam') #can change webcam to video or image ->
args.add_argument("--filePath", default=None) #None -> (filepath)

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    
    if args.mode in ["image"]:
        # read image
        img_path = './data/Shocked Black Guy.jpg'

        img = cv2.imread(img_path)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

        img = process_img(img, face_detection)
        
        # save image
        cv2.imwrite(os.path.join(output_dir, 'blurredFace.png'), img)
    
    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                        cv2.VideoWriter_fourcc(*'MP+V'), #codec
                                        25, #frame rate
                                        (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()
        
        cap.release()
        output_video.release()
    
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break
            ret, frame = cap.read()
        cap.release()