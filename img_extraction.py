from PIL import Image
from pathlib import Path
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image




class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)




if __name__== "__main__":
    
    fe = FeatureExtractor()
    for img_Path in sorted(Path("C:\\Users\\shilpa\\Downloads\\archive (1)\\img sample").glob("*.jpg")):
        print(img_Path)
        ##Feature extraction
        
        feature = fe.extract(img=Image.open(img_Path))
        print(type(feature), feature.shape)
       
        feature_path = Path("C:\\Users\\shilpa\\Downloads\\archive (1)\\feaures of img samples")/ (img_Path.stem+ ".npy")
        print(feature_path)
        
        ##Save the Feature
        np.save(feature_path, feature)
