from flask import Flask,request,render_template
import os

#custom package import
from Denoiser.autoencoder_denoiser import Denoiser

#initialising the app
app = Flask(__name__)


#index page
@app.route('/')
def index():
    return render_template('index.html',image={},style="display:None")


@app.route('/train',methods=['POST'])   
def train_autoencoder():
    if request.method == "POST":
        if request.json:
            data = request.json
            Denoiser.train_denoiser(data["epochs"],
                                    data["batch_size"],
                                    data["model_save_path"],
                                    data["graph_save_path"])
        else:
            return "Send Data Using Json!!"
    return "Training Completed!!"

@app.route('/denoise',methods=['POST'])
def denoise_image():
    if request.method == "POST":
        if request.json:
            data = request.json
            out = Denoiser.predict_denoiser(data["img_path"],
                                            data["model_path"],
                                        data["output_save_path"], 
                                        save_output=True)
            return "Output Saved"
        
        if request.files:
            image = request.files["image"]
            path= image.filename
            image.save(path)
            out = Denoiser.predict_denoiser(path,
                                            os.path.join("model","autoencoder.h5"),save_output=False)
            os.remove(path)
            return render_template('index.html',image=out,style={})
        
        
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) # for using port given by Heroku
    app.run(host="0.0.0.0",port=port)
    # app.run(debug=True)