from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)

dic = {0:'meningioma', 1:'pituitary',  2:'glioma', 3:'notumor'}

model = load_model('mymodel.h5')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(256,256))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 256,256,3)
    p = model.predict(i)
    return dic[np.argmax(p[0])]
   
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "brain_tumor detection..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		pr = predict_label(img_path)

	return render_template("index.html", prediction =pr, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = False)