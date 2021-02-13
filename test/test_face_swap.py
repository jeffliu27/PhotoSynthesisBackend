import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from Photo import Photo
from Album import Album

BLINK_PHOTO_PATH = os.path.join(os.path.dirname(__file__), './static/johnny_blink6.jpg')
NOBLINK_PHOTO_PATH = os.path.join(os.path.dirname(__file__), './static/johnny_noblink1.jpg')
# BLINK_PHOTO_PATH = os.path.join(os.path.dirname(__file__), './static/jeff_test/Jeff_blink.PNG')
# NOBLINK_PHOTO_PATH = os.path.join(os.path.dirname(__file__), './static/jeff_test/jeff_noblink.PNG')


my_album = Album()

blink_photo = Photo(BLINK_PHOTO_PATH)
noblink_photo = Photo(NOBLINK_PHOTO_PATH)

my_album.insert_photo(blink_photo)
my_album.insert_photo(noblink_photo)

my_album.facial_classification()
# blink_photo.blink_detect()
# noblink_photo.blink_detect()

my_album.update_base_photo_index(0)
my_album.blink_detection()
# person_id and photo_id of new face
# my_album.face_swap(swap_person_id = 0, newFace_photo_id = 1)

# search for non blinking face
# non_blinking_face = None

print(blink_photo.id_to_face)
print(noblink_photo.id_to_face)
# my_album.temp_face_swap(BLINK_PHOTO_PATH,NOBLINK_PHOTO_PATH)

my_album.remove_blinking_faces()