import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import tensorflow as tf
global graph,model

graph = tf.get_default_graph()

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: return render_template('index.html', label="No file")
		
		# read in file as raw pixels values
		# (ignore extra alpha channel and reshape as its a single image)
		img = misc.imread(file)
		#image_resized = misc.imresize(img, (28, 28))
		#img = img[:,:,:3]
		img1 = img[np.newaxis,:,:,:1]
		#img = img1.reshape(1, -1)

		# make prediction on new image
		with graph.as_default():
			prediction = model.predict(img1)
			results = np.argmax(prediction,axis = 1)
		print(prediction)
	
		# squeeze value from 1D array and convert to string for clean return
		label = str(np.squeeze(results))

		# switch for case where label=10 and number=0
		if label=='10': label='0'

		return render_template('index.html', label=label)


if __name__ == '__main__':
	# load ml model
	model = joblib.load('model.pkl')
	# start api
	app.run(host='192.168.43.17', port=8080, debug=True)
