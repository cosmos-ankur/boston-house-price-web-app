from flask import Flask , render_template , request , url_for
import pickle
import numpy as np 
model = pickle.load(open('boston_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def front():
	return render_template('front.html')

@app.route('/survey',methods=['POST'])
def main():
	name = request.form['name']
	email = request.form['email']
	
	return render_template('home.html',name=name,email=email)

@app.route('/predict',methods = ['POST'])
def home():
	crim = request.form['crim']
	zn = request.form['zn']
	indus = request.form['indus']
	chas = request.form['chas']
	nox = request.form['nox']
	rm = request.form['rm']
	age = request.form['age']
	dis = request.form['dis']
	rad = request.form['rad']
	tax = request.form['tax']
	ptr = request.form['ptr']
	b = request.form['b']
	lstat = request.form['lstat']

	arr = np.array([[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptr,b,lstat]])

	pred = model.predict(arr)
	return render_template('after.html',data=pred)

if __name__ == '__main__':
	app.run(debug=True)

