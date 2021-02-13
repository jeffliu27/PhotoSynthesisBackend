import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from Photo import Photo
TEST_PHOTO_PATH = os.path.join(os.path.dirname(__file__), './static/pic.jpg')

test_photo = Photo(TEST_PHOTO_PATH)
test_photo.face_detect()
# print(test_photo.id_to_face)
# print(test_photo.blinking_faces)

print("success")
