import os
import sys
import shutil
from flask import Flask, flash, request, redirect, url_for
from flask import render_template
from flask import send_from_directory
from flask import send_file
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_SESSIONS = 10

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from Photo import Photo
from Album import Album
from SessionManager import SessionManager

# commands: 
# set/export FLASK_APP=src/photosynthesis_webapp.py
# flask run

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create uploads folder
uploadFolder = os.path.join(os.path.dirname(__file__),app.config['UPLOAD_FOLDER'])

if os.path.exists(uploadFolder):
    shutil.rmtree(uploadFolder, ignore_errors=True)
os.mkdir(uploadFolder)

# initiate sessionManager
appSessions = SessionManager(MAX_SESSIONS)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def form():
    return render_template("form.html")

@app.route('/api/generate_blink_free', methods=['POST'])
def upload_file():
    # check for numFiles entry
    if 'numFiles' not in request.form or request.form['numFiles'] == "":
        content = {"Error": "Missing 'numFiles' in POST request"}
        return content, 400

    numFiles = int(request.form['numFiles'])

    if numFiles == 0:
        content = {"Error": "No files added in POST request"}
        return content, 400

    if 'basePhotoIndex' not in request.form or request.form['basePhotoIndex'] == "":
        content = {"Error": "Missing 'basePhotoIndex' in POST request"}
        return content, 400

    basePhotoIndex = int(request.form['basePhotoIndex'])

    # Error check
    for i in range(numFiles):
        entry = 'file['+str(i)+']'
        if entry not in request.files:
            content = {"Error": "Missing "+entry+" in POST request"}
            return content, 400

        file = request.files[entry]
        if file.filename == '':
            content = {"Error": "File field is empty in POST request"}
            return content, 400

        if not(file and allowed_file(file.filename)):
            content = {"Error": "File is wrong format (only accepts png, jpg, jpeg)"}
            return content, 400

    # Create a new session
    scale_percent = 0
    newAlbum = Album(scale_percent)
    new_session_id = appSessions.new_session(newAlbum)
    session_path = os.path.join(uploadFolder,new_session_id)
    os.mkdir(session_path)

    output_photo_path = ""
    # Save file in path
    for i in range(numFiles):
        entry = 'file['+str(i)+']'
        file = request.files[entry]
        filename = secure_filename(file.filename)

        if (i == basePhotoIndex):
            output_photo_path = os.path.join(session_path, "retouched_"+filename)
        file.save(os.path.join(session_path, filename))
        newAlbum.insert_photo(Photo(os.path.join(session_path, filename)))

    newAlbum.facial_classification()

    newAlbum.update_base_photo_index(basePhotoIndex)
    newAlbum.blink_detection()

    newAlbum.remove_blinking_faces()
    newAlbum.write_output_photo(output_photo_path)
    newAlbum.status = "READY"
    content = {"session_id": new_session_id}
    return content, 200

@app.route('/api/get_blink_free/<session_id>', methods=['GET'])
def fetch_file(session_id):
    # Error Check
    if session_id not in appSessions.sessions:
        content = {"Error": "session_id "+session_ids+" does not exist"}
        return content, 400

    if appSessions.sessions[session_id].status == "NOT READY":
        content = {"Error": "Photo for session_id "+session_ids+" is not ready"}
        return content, 400

    session_path = os.path.join(uploadFolder,session_id)
    album = appSessions.sessions[session_id]
    basePhotoIndex = album.base_photo_index
    filename = os.path.basename(album.photos[basePhotoIndex].img_path)
    filePath = os.path.join(session_path, "retouched_" + filename)

    if filePath is None:
        content = {"Error": "Photo at filepath "+filePath+" is missing"}
        return content, 400
    else:
        print("sending " + filePath)
        return send_file(filePath, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

