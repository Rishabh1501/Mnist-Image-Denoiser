from flask import Flask,request

#custom package import
from Denoiser.autoencoder_denoiser import Denoiser

#initialising the app
app = Flask(__name__)

@app.route('/train',methods=['POST'])   
def train_autoencoder():
    if request.method == "POST":
        data = request.json
        Denoiser.train_denoiser(data["epochs"],
                                data["batch_size"],
                                data["model_save_path"],
                                data["graph_save_path"])
        
    return "Training Completed"

@app.route('/denoise',methods=['POST'])
def denoise_image():
    if request.method == "POST":
        data = request.json
        out = Denoiser.predict_denoiser(data["img_path"],
                                        data["model_path"],
                                        data["output_save_path"])
        
        return "Output Saved"
        
if __name__ == "__main__":
    app.run(debug=True)