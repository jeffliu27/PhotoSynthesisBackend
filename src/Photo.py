import cv2
import numpy as np
import dlib
import math
import os

PREDICTOR_68_PATH = os.path.join(os.path.dirname(__file__), "./resources/shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def blink_detect_helper(l, landmarks):
    left_point = (landmarks.part(l[0]).x, landmarks.part(l[0]).y)
    right_point = (landmarks.part(l[3]).x, landmarks.part(l[3]).y)    

    center_top = midpoint(landmarks.part(l[1]), landmarks.part(l[2]))
    center_bottom = midpoint(landmarks.part(l[-1]), landmarks.part(l[-2]))

    # get left eye ratio
    hor_line_lenght = math.hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    eye_ratio = hor_line_lenght / ver_line_lenght

    # hor_line = cv2.line(img, left_point, right_point, (0, 255, 0), 1)
    # ver_line = cv2.line(img, center_top, center_bottom, (0, 255, 0), 1)

    return eye_ratio

class Photo:
    def __init__(self, _img_path, _scale_percent = 50):
        self.img_path = _img_path # file path in s3
        self.blinking_faces = set() # set(people_ids)
        self.id_to_face = {} #id -> (face_rect, face_landmarks)
        self.scale_percent = _scale_percent

    def blink_detect_per_face(self, landmarks):
        # takes self.id_to_face and populates blinking_faces with the faces that are blinking
        left_eye_ratio = blink_detect_helper([36,37,38,39,40,41], landmarks)
        right_eye_ratio = blink_detect_helper([42,43,44,45,46,47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.2:
            return True

        return False

    def face_detect(self):
        img = cv2.imread(self.img_path)

        assert(img is not None and img.size != 0)
        print("Running Facial Detection on Img:{}".format(self.img_path))

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_68_PATH)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert pic to gray frame -> lighter image save computational power
        faces = detector(gray_img) # facial detection

        landmarks = []

        for face in faces:
            # # only for demo purposes - draws a rectangle around the face
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 5)

            landmark = predictor(gray_img, face)
            landmarks.append(landmark)
        
        # #demo
        
        # width = int(img.shape[1] * self.scale_percent / 100)
        # height = int(img.shape[0] * self.scale_percent / 100)
        # resized_final = cv2.resize(img, (width,height))
        # cv2.imshow("faceDetection", resized_final)    
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return faces, landmarks
    
    # run after facial classification
    def blink_detect(self):
        # TODO REMOVE LATER
        img = cv2.imread(self.img_path)
        for id, face in self.id_to_face.items():
            (_, landmarks) = face

            if self.blink_detect_per_face(landmarks):
                cv2.putText(img, "{} blinked".format(id), (landmarks.part(3).x, landmarks.part(9).y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)
                self.blinking_faces.add(id)
            else:
                cv2.putText(img, "{}".format(id), (landmarks.part(3).x, landmarks.part(9).y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 5)
        # move this somewhere else later

        if (self.scale_percent):
            width = int(img.shape[1] * self.scale_percent / 100)
            height = int(img.shape[0] * self.scale_percent / 100)
            resized_final = cv2.resize(img, (width,height))

            cv2.imshow("blinkDetection", resized_final)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
        return


    # def face_blink_detect(self):
    #     img = cv2.imread(self.img_path)

    #     detector = dlib.get_frontal_face_detector()
    #     predictor = dlib.shape_predictor(PREDICTOR_68_PATH)

    #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert pic to gray frame -> lighter image save computational power
    #     faces = detector(gray_img) # facial detection

    #     SOME_ID_FROM_CLASSIFICATION = 0

    #     for face in faces:
    #         x, y = face.left(), face.top()
    #         x1, y1 = face.right(), face.bottom()
    #         cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

    #         landmarks = predictor(gray_img, face)
    #         # get id from facial classification
    #         # Shouldn't be two faces of the same person in one photo 
    #         # rip twins
    #         assert(SOME_ID_FROM_CLASSIFICATION not in self.id_to_face)
    #         self.id_to_face[SOME_ID_FROM_CLASSIFICATION] = (face, landmarks)
    #         if self.blink_detect_per_face(landmarks):
    #             cv2.putText(img, "{} blinked".format(SOME_ID_FROM_CLASSIFICATION), (landmarks.part(3).x, landmarks.part(9).y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    #             self.blinking_faces.add(SOME_ID_FROM_CLASSIFICATION)
    #         SOME_ID_FROM_CLASSIFICATION += 1
        
    #     cv2.imshow("faceDetection", img)
    #     key = cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        # return