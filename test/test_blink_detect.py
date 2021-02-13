import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from Photo import Photo
from Album import Album

GROUP_PHOTO_1_PATH = os.path.join(os.path.dirname(__file__), './static/pic.jpg')


my_album = Album()

group1_photo = Photo(GROUP_PHOTO_1_PATH)


my_album.insert_photo(group1_photo)

my_album.facial_classification()

group1_photo.blink_detect()
