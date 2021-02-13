import cv2
import numpy as np
import dlib
import time
import os
import math
from imutils import face_utils
from sklearn.cluster import KMeans

PREDICTOR_68_PATH = os.path.join(os.path.dirname(__file__), "./resources/shape_predictor_68_face_landmarks.dat")

DEBUGGING = False

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def delaunay_triangulation(landmarks_points):
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)

    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

        # DEBUGGING: Draw Triangles 
        # cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        # cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        # cv2.line(img, pt1, pt3, (0, 0, 255), 2)

    return indexes_triangles

# Description: 
    # Groups list of Photo class objects together for easy access when performing face swap.
class Album:
    def __init__(self, _scale_percent = 50):
        self.photos = []
        self.base_photo_index = 0
        self.output_photo = None
        self.status = "NOT READY"
        self.scale_percent = _scale_percent

        self.k_mean_model = None
    def insert_photo(self,newPhoto):
        newPhoto.scale_percent = self.scale_percent
        self.photos.append(newPhoto)
    

    def write_output_photo(self, path):
        cv2.imwrite(path, self.output_photo)
    # Changes the index which identifies the base photo.
    # Picked by the user. Default 0 if no new index is passed in.
    def update_base_photo_index(self, new_index = 0):
        self.base_photo_index = new_index
        self.output_photo = cv2.imread(self.photos[self.base_photo_index].img_path)

    def blink_detection(self):
        for photo in self.photos:
            photo.blink_detect()
        return
    def get_landmarks(self, cv2_img, face_coord):
        # face coord is a dlib face object

        predictor = dlib.shape_predictor(PREDICTOR_68_PATH)

        gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) # convert pic to gray frame -> lighter image save computational power

        landmark = predictor(gray_img, face_coord)
        return landmark

    # Given the id of a blinking person, find 
    def find_alternate_photo(self, blinking_person, searched_photos):
        for photo_id, photo in enumerate(self.photos):
            if photo_id != self.base_photo_index and photo_id not in searched_photos:
                print("     Searching for person {} in photo {}".format(blinking_person, photo_id))
                if blinking_person not in photo.blinking_faces and blinking_person in photo.id_to_face.keys():
                    return photo_id
        return None

    def remove_blinking_faces(self):
        base_photo = self.photos[self.base_photo_index]

        for blinking_person in base_photo.blinking_faces:
            
            valid_face_swap = False
            searched_photos = set()
            while not valid_face_swap:

                alternate_photo_id = self.find_alternate_photo(blinking_person, searched_photos)
                
                if (alternate_photo_id is not None):
                    print("Found a potential replacement face for person {} in photo number{}!".format(blinking_person, alternate_photo_id))
                    # face swap with alternate photo, and temporary set to valid
                    swapped_photo = self.face_swap(blinking_person, alternate_photo_id)
                    valid_face_swap = True
                    
                    # check if face swap produced valid results
                    blinking_face_coord = self.photos[self.base_photo_index].id_to_face[blinking_person][0]
                    landmark = self.get_landmarks(swapped_photo, blinking_face_coord)
                    # blink test
                    if (self.photos[self.base_photo_index].blink_detect_per_face(landmark)):
                        print("         Person is still blinking")
                        valid_face_swap = False
                    original_landmark = self.photos[self.base_photo_index].id_to_face[blinking_person][1]

                    landmark_np = face_utils.shape_to_np(landmark)

                    x, y = landmark_np.shape
                    id_post_face_swap = self.k_mean_model.predict(landmark_np.reshape(-1, x * y))
                    
                    # face similarity heuristic:
                    # nose len can't be 10% bigger
                    # mouth width can't be 10% bigger
                    top_nose = (original_landmark.part(28).x, original_landmark.part(28).y)
                    bot_nose = (original_landmark.part(34).x, original_landmark.part(34).y)

                    top_mouth = (original_landmark.part(49).x, original_landmark.part(49).y)
                    bot_mouth = (original_landmark.part(55).x, original_landmark.part(55).y)
                    
                    nose_offset_orig = math.hypot(top_nose[0]-bot_nose[0], top_nose[1]-bot_nose[1])
                    mouth_offset_orig = math.hypot(top_mouth[0]-bot_mouth[0], top_mouth[1]-bot_mouth[1])



                    top_swap = (landmark.part(28).x, landmark.part(28).y)
                    bot_swap = (landmark.part(34).x, landmark.part(34).y)
                    top_mouth_swap = (landmark.part(49).x, landmark.part(49).y)
                    bot_mouth_swap = (landmark.part(55).x, landmark.part(55).y)

                    nose_offset_swap = math.hypot(top_swap[0]-bot_swap[0], top_swap[1]-bot_swap[1])
                    mouth_offset_swap = math.hypot(top_mouth_swap[0]-bot_mouth_swap[0], top_mouth_swap[1]-bot_mouth_swap[1])


                    if (abs(nose_offset_orig - nose_offset_swap) > 0.1 * nose_offset_orig):
                        print("         nose len bigger than usual")
                        valid_face_swap = False

                    
                    if (abs(mouth_offset_orig - mouth_offset_swap) > 0.1 * mouth_offset_orig):
                        print("         mouth width bigger than usual")
                        valid_face_swap = False


                    # classfication test
                    if (id_post_face_swap != blinking_person):
                        print("         Not recognizable as the same person")
                        valid_face_swap = False
                    
                    if valid_face_swap:
                        self.output_photo = swapped_photo
                    else:
                        # add to the searched photo set 
                        searched_photos.add(alternate_photo_id)
                        print("This swap didnt qualify, looking for another photo to swap with...")
                else:
                    print("Couldn't find a successful replacement for person {} :(".format(blinking_person))
                    break

    def face_swap(self, swap_person_id, newFace_photo_id):
        # Get the Base Photos and gray scale versions
        baseFace_img = self.output_photo
        baseFace_img_gray = cv2.cvtColor(baseFace_img, cv2.COLOR_BGR2GRAY)

        newFace_img_path = self.photos[newFace_photo_id].img_path
        newFace_img = cv2.imread(newFace_img_path)
        newFace_img_gray = cv2.cvtColor(newFace_img, cv2.COLOR_BGR2GRAY)

        # Get face locations
        baseFace_landmarks = self.photos[self.base_photo_index].id_to_face[swap_person_id][1]
        newFace_landmarks = self.photos[newFace_photo_id].id_to_face[swap_person_id][1]

        ####################################################
        #
        # New Face
        #
        ####################################################
        newFace_landmarks_points = []
        eyeOffsets = {}
        for n in range(0, 68):
            x = newFace_landmarks.part(n).x
            y = newFace_landmarks.part(n).y
            newFace_landmarks_points.append((x, y))

        # Right eye from Left
        # eyeOffsets[36] = tuple(np.subtract(newFace_landmarks_points[42],newFace_landmarks_points[36]))

        # # Left Eye
        # eyeOffsets[37] = tuple(np.subtract(newFace_landmarks_points[37],newFace_landmarks_points[36]))
        # eyeOffsets[41] = tuple(np.subtract(newFace_landmarks_points[41],newFace_landmarks_points[36]))
        # eyeOffsets[38] = tuple(np.subtract(newFace_landmarks_points[38],newFace_landmarks_points[36]))
        # eyeOffsets[40] = tuple(np.subtract(newFace_landmarks_points[40],newFace_landmarks_points[36]))
        # eyeOffsets[39] = tuple(np.subtract(newFace_landmarks_points[39],newFace_landmarks_points[36]))

        # # Right Eye
        # eyeOffsets[43] = tuple(np.subtract(newFace_landmarks_points[43],newFace_landmarks_points[42]))
        # eyeOffsets[47] = tuple(np.subtract(newFace_landmarks_points[47],newFace_landmarks_points[42]))
        # eyeOffsets[44] = tuple(np.subtract(newFace_landmarks_points[44],newFace_landmarks_points[42]))
        # eyeOffsets[46] = tuple(np.subtract(newFace_landmarks_points[46],newFace_landmarks_points[42]))
        # eyeOffsets[45] = tuple(np.subtract(newFace_landmarks_points[45],newFace_landmarks_points[42]))

        # New offsets
        line_p1 = np.asarray(newFace_landmarks_points[36],dtype=np.float64)
        line_p2 = np.asarray(newFace_landmarks_points[39],dtype=np.float64)

        if DEBUGGING:
            newFace_img = cv2.line(newFace_img, (int(line_p1[0]),int(line_p1[1])),(int(line_p2[0]),int(line_p2[1])), color=(255, 0, 0), thickness=1)

        # left eye
        for i in range(36,42):
            if i not in [36,39]:
                curr_point = np.asarray(newFace_landmarks_points[i],dtype=np.float64)
                point_p1 = curr_point - line_p1
                line = line_p2 - line_p1
                closestPointToLine = line_p1 + np.dot(point_p1,line) / np.dot(line,line) * line

                direction = np.cross(curr_point,closestPointToLine)
                if direction > 0:
                    eyeOffsets[i] = np.linalg.norm(np.subtract(curr_point,closestPointToLine))
                else:
                    eyeOffsets[i] = np.linalg.norm(np.subtract(curr_point,closestPointToLine))

                if DEBUGGING:
                    p1 = (int(newFace_landmarks_points[i][0]),int(newFace_landmarks_points[i][1]))
                    p2 = (int(closestPointToLine[0]),int(closestPointToLine[1]))
                    newFace_img = cv2.line(newFace_img, p1, p2, color=(0, 255, 0), thickness=1)


        # right eye
        line_p1 = np.asarray(newFace_landmarks_points[42],dtype=np.float64)
        line_p2 = np.asarray(newFace_landmarks_points[45],dtype=np.float64)

        if DEBUGGING:
            newFace_img = cv2.line(newFace_img, (int(line_p1[0]),int(line_p1[1])),(int(line_p2[0]),int(line_p2[1])), color=(255, 0, 0), thickness=1)
        for i in range (42,48):
            if i not in [42,45]:
                curr_point = np.asarray(newFace_landmarks_points[i],dtype=np.float64)
                point_p1 = curr_point - line_p1
                line = line_p2 - line_p1
                closestPointToLine = line_p1 + np.dot(point_p1,line) / np.dot(line,line) * line

                direction = np.cross(curr_point,closestPointToLine)
                if direction > 0:
                    eyeOffsets[i] = np.linalg.norm(np.subtract(curr_point,closestPointToLine))
                else:
                    eyeOffsets[i] = np.linalg.norm(np.subtract(curr_point,closestPointToLine))

                #     p1 = (int(newFace_landmarks_points[i][0]),int(newFace_landmarks_points[i][1]))
                #     p2 = (int(closestPointToLine[0]),int(closestPointToLine[1]))
                #     newFace_img = cv2.line(newFace_img, p1, p2, color=(0, 255, 0), thickness=1)

        # for i in range(36,48):
            # newFace_img = cv2.circle(newFace_img, (int(newFace_landmarks_points[i][0]),int(newFace_landmarks_points[i][1])), radius=0, color=(0, 0, 255), thickness=1)

        # Delaunay triangulation
        newFace_triangles = delaunay_triangulation(newFace_landmarks_points)

        ####################################################
        #
        # Base Face
        #
        ####################################################
        baseFace_landmarks_points = []
        for n in range(0, 68):
            x = baseFace_landmarks.part(n).x
            y = baseFace_landmarks.part(n).y
            baseFace_landmarks_points.append((x, y))

        # Right eye from Left
        # baseFace_landmarks_points[42] = tuple(np.add(baseFace_landmarks_points[36],eyeOffsets[36]))

        # # Left Eye
        # baseFace_landmarks_points[37] = tuple(np.add(baseFace_landmarks_points[36],eyeOffsets[37]))
        # baseFace_landmarks_points[41] = tuple(np.add(baseFace_landmarks_points[36],eyeOffsets[41]))
        # baseFace_landmarks_points[38] = tuple(np.add(baseFace_landmarks_points[36],eyeOffsets[38]))
        # baseFace_landmarks_points[40] = tuple(np.add(baseFace_landmarks_points[36],eyeOffsets[40]))
        # baseFace_landmarks_points[39] = tuple(np.add(baseFace_landmarks_points[36],eyeOffsets[39]))

        # # Right Eye
        # baseFace_landmarks_points[43] = tuple(np.add(baseFace_landmarks_points[42],eyeOffsets[43]))
        # baseFace_landmarks_points[47] = tuple(np.add(baseFace_landmarks_points[42],eyeOffsets[47]))
        # baseFace_landmarks_points[44] = tuple(np.add(baseFace_landmarks_points[42],eyeOffsets[44]))
        # baseFace_landmarks_points[46] = tuple(np.add(baseFace_landmarks_points[42],eyeOffsets[46]))
        # baseFace_landmarks_points[45] = tuple(np.add(baseFace_landmarks_points[42],eyeOffsets[45]))

        #left eye
        line_p1 = np.asarray(baseFace_landmarks_points[36], dtype=np.float64)
        line_p2 = np.asarray(baseFace_landmarks_points[39], dtype=np.float64)

        if DEBUGGING:
            baseFace_img = cv2.line(baseFace_img, (int(line_p1[0]),int(line_p1[1])),(int(line_p2[0]),int(line_p2[1])), color=(255, 0, 0), thickness=1)

        for i in range(36,42):
            if i not in [36,39]:
                curr_point = np.asarray(baseFace_landmarks_points[i],dtype=np.float64)
                point_p1 = curr_point - line_p1
                line = line_p2 - line_p1
                closestPointToLine = line_p1 + np.dot(point_p1,line) / np.dot(line,line) * line

                curr_dist = np.linalg.norm(np.subtract(curr_point,closestPointToLine))
                unitvector_offset = (curr_point - closestPointToLine) / curr_dist
                vector_offset = unitvector_offset * eyeOffsets[i]

                (new_x, new_y) = closestPointToLine + vector_offset
                baseFace_landmarks_points[i] = (int(new_x),int(new_y))

                if DEBUGGING:
                    p1 = (int(baseFace_landmarks_points[i][0]),int(baseFace_landmarks_points[i][1]))
                    p2 = (int(closestPointToLine[0]),int(closestPointToLine[1]))
                    baseFace_img = cv2.line(baseFace_img, p1, p2, color=(0, 255, 0), thickness=1)

        line_p1 = np.asarray(baseFace_landmarks_points[42], dtype=np.float64)
        line_p2 = np.asarray(baseFace_landmarks_points[45], dtype=np.float64)

        if DEBUGGING:
            baseFace_img = cv2.line(baseFace_img, (int(line_p1[0]),int(line_p1[1])),(int(line_p2[0]),int(line_p2[1])), color=(255, 0, 0), thickness=1)
        for i in range(42,48):
            if i not in [42,45]:
                curr_point = np.asarray(baseFace_landmarks_points[i],dtype=np.float64)
                point_p1 = curr_point - line_p1
                line = line_p2 - line_p1
                closestPointToLine = line_p1 + np.dot(point_p1,line) / np.dot(line,line) * line

                curr_dist = np.linalg.norm(np.subtract(curr_point,closestPointToLine))
                unitvector_offset = (curr_point - closestPointToLine) / curr_dist
                vector_offset = unitvector_offset * eyeOffsets[i]

                (new_x, new_y) = closestPointToLine + vector_offset
                baseFace_landmarks_points[i] = (int(new_x),int(new_y))

                # p1 = (int(baseFace_landmarks_points[i][0]),int(baseFace_landmarks_points[i][1]))
                # p2 = (int(closestPointToLine[0]),int(closestPointToLine[1]))
                # baseFace_img = cv2.line(baseFace_img, p1, p2, color=(0, 255, 0), thickness=1)

        # for i in range(36,40):
        #     baseFace_img = cv2.circle(baseFace_img, (int(baseFace_landmarks_points[i][0]),int(baseFace_landmarks_points[i][1])), radius=0, color=(0, 0, 255), thickness=1)

        baseFace_points = np.array(baseFace_landmarks_points, np.int32)
        baseFace_convexhull = cv2.convexHull(baseFace_points)

        ####################################################
        #
        # Triangulation of New Face onto Base Face
        #
        ####################################################
        lines_space_mask = np.zeros_like(newFace_img_gray)

        baseFace_img_height, baseFace_img_width, baseFace_img_channels = baseFace_img.shape
        newFace = np.zeros((baseFace_img_height, baseFace_img_width, baseFace_img_channels), np.uint8)

        for triangle in newFace_triangles:
            # Triangulation of the first face
            tr1_pt1 = newFace_landmarks_points[triangle[0]]
            tr1_pt2 = newFace_landmarks_points[triangle[1]]
            tr1_pt3 = newFace_landmarks_points[triangle[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = newFace_img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)

            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Lines space
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
            cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
            lines_space = cv2.bitwise_and(newFace_img, newFace_img, mask=lines_space_mask) # maybe dont need

            # Triangulation of second face
            tr2_pt1 = baseFace_landmarks_points[triangle[0]]
            tr2_pt2 = baseFace_landmarks_points[triangle[1]]
            tr2_pt3 = baseFace_landmarks_points[triangle[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            newFace_rect_area = newFace[y: y + h, x: x + w]
            newFace_rect_area_gray = cv2.cvtColor(newFace_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(newFace_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            newFace_rect_area = cv2.add(newFace_rect_area, warped_triangle)
            newFace[y: y + h, x: x + w] = newFace_rect_area

        ####################################################
        #
        # Swap Faces
        #
        ####################################################
        baseFace_face_mask = np.zeros_like(baseFace_img_gray)
        baseFace_head_mask = cv2.fillConvexPoly(baseFace_face_mask, baseFace_convexhull, 255)
        baseFace_face_mask = cv2.bitwise_not(baseFace_head_mask)

        baseFace_background = cv2.bitwise_and(baseFace_img, baseFace_img, mask=baseFace_face_mask)
        result = cv2.add(baseFace_background, newFace)

        (x, y, w, h) = cv2.boundingRect(baseFace_convexhull)
        baseFace_center = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, baseFace_img, baseFace_head_mask, baseFace_center, cv2.NORMAL_CLONE)

        ####################################################
        #
        # Display Faces
        #
        ####################################################
        if (self.scale_percent):
            width = int(seamlessclone.shape[1] * self.scale_percent / 100)
            height = int(seamlessclone.shape[0] * self.scale_percent / 100)
            resized_final = cv2.resize(seamlessclone, (width,height))

            print("Face swapped person {} from base photo with photo {}".format(swap_person_id, newFace_photo_id))

            cv2.imshow("no blinking original",cv2.resize(newFace_img,(int(newFace_img.shape[1]* self.scale_percent / 100), int(newFace_img.shape[0]* self.scale_percent / 100))))
            # cv2.imshow("blinking original",cv2.resize(baseFace_img,(int(baseFace_img.shape[1]* self.scale_percent / 100),int(baseFace_img.shape[0]* self.scale_percent / 100))))
            # cv2.imshow("face swap initial result",cv2.resize(result,(int(result.shape[1]* self.scale_percent / 100), int(result.shape[0]* self.scale_percent / 100))))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imshow("blended face swap result", cv2.resize(seamlessclone,(int(seamlessclone.shape[1]* self.scale_percent / 100),int(seamlessclone.shape[0]* self.scale_percent / 100))))

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return seamlessclone

    def facial_classification(self):
        num_faces = 0
        face_to_photo = []
        facial_landmarks = []

        for i, photo in enumerate(self.photos):
            faces, landmarks = photo.face_detect()
            num_faces = max(num_faces, len(faces))

            for j, face in enumerate(faces):
                face_to_photo.append((i, face, landmarks[j]))

            for landmark in landmarks:
                landmarks_np = face_utils.shape_to_np(landmark)
                facial_landmarks.append(landmarks_np)

        facial_landmarks = np.asarray(facial_landmarks)
        nsamples, nx, ny = facial_landmarks.shape
        landmarks = facial_landmarks.reshape((nsamples,nx*ny))

        self.k_mean_model = KMeans(n_clusters=num_faces)
        self.k_mean_model.fit(landmarks)
        labels = self.k_mean_model.labels_

        for i, label in enumerate(labels):
            (id, face, landmark) = face_to_photo[i]
            # no twins pls
            self.photos[id].id_to_face[label] = (face, landmark)