from flask import Flask,render_template,redirect,url_for,Response, jsonify
import cv2
import pickle
from flask import request
import os
import numpy as np

pca=pickle.load(open("FFHQ_pca.pickle",'rb'))
scaler=pickle.load(open("FFHQ_scaler.pickle",'rb'))
model=pickle.load(open("FFHQ_model.pickle",'rb'))
encoder=pickle.load(open("FFHQ_encoder.pickle",'rb'))

cam=cv2.VideoCapture(0)
app=Flask(__name__,static_url_path='/static')
UPLOAD_FOLDER='templates/static/images/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.static_folder='stat'

flag=0
def camread():
    while True:
        success,frame = cam.read()
        if not success :
            break
        elif flag==0:
            ret, buf = cv2.imencode('.jpg', frame)
            frame = buf.tobytes()

            yield (b'--frame\r\n' b'Content-Type" image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def camcapture():
        flag=1
        success,frame = cam.read()
        a=cv2.imwrite('dataimage.jpg', frame)
        print(a)
        
def get_pred(i):
    transformed=pca.transform([np.ravel(i)])
    scaled=scaler.transform(transformed)
    prediction=model.predict(scaled)
    return encoder.inverse_transform(prediction)[0]


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/choose',methods=['POST'])
def choose():
    return render_template('index.html')

@app.route('/snap',methods=['POST'])
def snap():
    return render_template('snap.html')

@app.route('/snap2')
def snap2():
    return Response(camread(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click',methods=['POST'])
def click():
    flag=1
    Response(camcapture(),mimetype='multipart/x-mixed-replace; boundary=frame')
    img=cv2.imread("dataimage.jpg",cv2.IMREAD_COLOR)
    img=cv2.resize(img,(128,128))
    ans=get_pred(img)
    ans=ans.upper()

    return render_template('output.html',img1=ans,img2="temp.jpeg")


@app.route('/click2')
def click2():
    flag=1
    Response(camread(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['image']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # file.save("C:/Users/admin/Documents/CSE at Nirma University/5th Semester/Innovative/ML/static",filename)
    img=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename),cv2.IMREAD_COLOR)
    img=cv2.resize(img,(128,128))
    cv2.imwrite("C:/Users/admin/Documents/CSE at Nirma University/5th Semester/Innovative/ML/static/temp.jpeg",img)

    # cv2.imwrite("C:/Users/admin/Documents/CSE at Nirma University/5th Semester/Innovative/ML/templates/static/dataimg.jpeg",file)
    ans=get_pred(img)
    ans=ans.upper()

    return render_template('output.html',img1=ans,img2="temp.jpeg")



if __name__=='__main__':
    app.run(debug=True)




