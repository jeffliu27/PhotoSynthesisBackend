import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from Photo import Photo
from Album import Album

GROUP_PHOTO_1_PATH = os.path.join(os.path.dirname(__file__), './static/group1.jpg')
GROUP_PHOTO_2_PATH = os.path.join(os.path.dirname(__file__), './static/group2.jpg')
GROUP_PHOTO_3_PATH = os.path.join(os.path.dirname(__file__), './static/group3.jpg')

my_album = Album()

group1_photo = Photo(GROUP_PHOTO_1_PATH)
group2_photo = Photo(GROUP_PHOTO_2_PATH)
group3_photo = Photo(GROUP_PHOTO_3_PATH)

my_album.insert_photo(group1_photo)
my_album.insert_photo(group2_photo)
my_album.insert_photo(group3_photo)

my_album.facial_classification()

print(group1_photo.id_to_face)
print(group2_photo.id_to_face)
print(group3_photo.id_to_face)

assert(group1_photo.id_to_face.keys()==group2_photo.id_to_face.keys()==group3_photo.id_to_face.keys())

#todo: test facial classification where different people change