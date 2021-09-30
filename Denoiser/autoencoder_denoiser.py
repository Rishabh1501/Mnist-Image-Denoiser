from ConvAutoencoder.convautoencoder import ConvAutoEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Denoiser():
    
    @staticmethod
    def train_denoiser(epochs,batch_size,model_save_path,graph_save_path):
        print("[INFO] loading MNIST dataset...")
        ((trainX, _), (testX, _)) = mnist.load_data()
        # add a channel dimension to every image in the dataset, then scale
        # the pixel intensities to the range [0, 1]
        trainX = np.expand_dims(trainX, axis=-1)
        testX = np.expand_dims(testX, axis=-1)
        trainX = trainX.astype("float32") / 255.0
        testX = testX.astype("float32") / 255.0
        
        # sample noise from a random normal distribution centered at 0.5 (since
        # our images lie in the range [0, 1]) and a standard deviation of 0.5
        trainNoise = np.random.normal(loc=0.5, scale=0.5, size=trainX.shape)
        testNoise = np.random.normal(loc=0.5, scale=0.5, size=testX.shape)
        trainXNoisy = np.clip(trainX + trainNoise, 0, 1)
        testXNoisy = np.clip(testX + testNoise, 0, 1)

        # construct our convolutional autoencoder
        print("[INFO] building autoencoder...")
        (encoder, decoder, autoencoder) = ConvAutoEncoder.build(28, 28, 1)
        opt = Adam(lr=1e-3)
        autoencoder.compile(loss="mse", optimizer=opt)
        # train the convolutional autoencoder
        H = autoencoder.fit(
            trainXNoisy, trainX,
            validation_data=(testXNoisy, testX),
            epochs=epochs,
            batch_size=batch_size)
        # construct a plot that plots and saves the training history
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(graph_save_path)
        autoencoder.save(model_save_path, save_format="h5")
    
    @staticmethod
    def predict_denoiser(img_path,model_path,output_save_path):
        #preprocessing the image
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        img = img.astype("float32") / 255.0
        img = np.reshape(img,(28,28,1))
        img = np.expand_dims(img,axis=0)
        
        #loading the model
        model = load_model(model_path)
        #prediction
        predict = model.predict(img)
        original = (img[0] * 255).astype("uint8")
        output = (predict[0]*255).astype("uint8")
        
        cv2.imwrite(output_save_path,np.hstack([original,output]))
        return output
        