from PIL import Image
from pathlib import Path
import numpy as np
from datetime import datetime
from flask import Flask, render_template


app = Flask(__name__)

#read img features
fe = FeatureExtractor()
features = []
img_path = []

for feature_path in Path("C:\\Users\\shilpa\\Downloads\\archive (1)\\feaures of img samples").glob("*.npy"):
    features.append(np.load(feature_path))
    img_path.append(Path("C:\\Users\\shilpa\\Downloads\\archive (1)\\img sample") / (feature_path.stem + ".jpg"))
features = np.array(features)    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query_img"]
        
        #Save query img
        img = Image.open(file.stream) #PIL IMG
        uploaded_img_path = "C:\\Users\\shilpa\\Downloads\\archive (1)\\uploaded" + datetime.now().isoformat().replace(":", ".")+ "_" +file.filename
        img.save(uploaded_img_path)
        
        ##Runtime search
        query = fe.extract(img)
        dists = np.linalg.norm(feature - query, axis = 1) #L2 distance
        ids  = np.argsort(distance)[:10]
        scores = [(dists[id], img_paths[id]) for i in ids]

        return render_template("index.html", query_path=uploaded_img_path)
    else:
        return render_template("index.html")

               
    print(scores)           
    


if __name__=="__main__":
    app.run()
    