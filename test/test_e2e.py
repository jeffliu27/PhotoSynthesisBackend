import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from Photo import Photo
from Album import Album

# PHOTO_PATH_1 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/charlie_blink.jpg')
PHOTO_PATH_2 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/charlie_eugene_blink.jpg')
PHOTO_PATH_3 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/cynthia_half_blinking.jpg')
PHOTO_PATH_4 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/erick_blink.jpg')
PHOTO_PATH_5 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/erick_katie_blink.jpg')
PHOTO_PATH_6 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/jeff_blink.jpg')
PHOTO_PATH_7 = os.path.join(os.path.dirname(__file__), './static/set1_cannonball2017/no_one_blink.jpg')

scale_percent = 20

if len(sys.argv) > 1:
    scale_percent = int(sys.argv[1])

my_album = Album(scale_percent)
# my_album.insert_photo(Photo(PHOTO_PATH_1))
my_album.insert_photo(Photo(PHOTO_PATH_2))
my_album.insert_photo(Photo(PHOTO_PATH_3))
my_album.insert_photo(Photo(PHOTO_PATH_4))
my_album.insert_photo(Photo(PHOTO_PATH_5))
my_album.insert_photo(Photo(PHOTO_PATH_6))
my_album.insert_photo(Photo(PHOTO_PATH_7))


my_album.facial_classification()

# case1: Jeff blinking
my_album.update_base_photo_index(4)
my_album.blink_detection()

my_album.remove_blinking_faces()

# # Future test cases below
# my_album.update_base_photo_index(0)
# my_album.remove_blinking_faces()

# my_album.update_base_photo_index(2)
# my_album.remove_blinking_faces()

# my_album.update_base_photo_index(1)
# my_album.remove_blinking_faces()

# my_album.update_base_photo_index(3)
# my_album.remove_blinking_faces()

# my_album.update_base_photo_index(4)
# my_album.remove_blinking_faces()

# my_album.update_base_photo_index(6)
# my_album.remove_blinking_faces()

