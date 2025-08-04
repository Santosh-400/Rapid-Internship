from flask import Flask, render_template_string, request
import numpy as np
import joblib
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ==== Flask & Model Setup ====
app = Flask(__name__)

MODEL_PATH = 'fake_account_xgb.pkl'
SCALER_PATH = 'scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten()).to(device)
resnet.eval()
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_img_features(img):
    img = img.convert("RGB")
    tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor).cpu().numpy().flatten()
    return feat

# ==== HTML Template ====
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake Account Detection</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial,sans-serif; background:#f7f9fa; padding:40px;}
        form { background:#fff;max-width:480px;margin:0 auto;padding:24px 40px 18px 40px;border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.10);}
        h2{text-align:center;}
        label, input, textarea { display:block; width:100%; margin-bottom:14px;font-size:1rem;}
        button {background:#3b82f6; color:#fff;font-size:1rem;border:0;padding:10px 18px;
            border-radius:4px;cursor:pointer; margin-top:8px;}
        .result {padding:14px 0 3px 0; border-radius:4px;font-weight:bold;text-align:center;}
        .real {background:#e0f7fa;color:#0288d1;}
        .fake {background:#ffdede;color:#d40000;}
        .error {background:#ffdede;color:#d40000;}
    </style>
</head>
<body>
    <form method="post" action="/" enctype="multipart/form-data">
        <h2>Fake Account Detection</h2>

        <label>Verified Account?</label>
        <select name="verified" required>
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>

        <label>Followers Count</label>
        <input type="number" name="followers" min="0" required>

        <label>Following Count</label>
        <input type="number" name="follows" min="0" required>

        <label>Biography</label>
        <textarea name="biography" rows="3" required></textarea>

        <label>Profile Image</label>
        <input type="file" name="profile_img" accept="image/*" required>

        <button type="submit">Check Account</button>
        {% if result %}
            <div class="result {{result_class}}">Prediction: {{result|upper}}</div>
        {% elif error %}
            <div class="result error">{{error}}</div>
        {% endif %}
    </form>
</body>
</html>
"""

# ==== Main Route ====
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    result_class = ""

    if request.method == "POST":
        try:
            # Process fields
            verified = int(request.form["verified"])
            followers = float(request.form["followers"])
            follows = float(request.form["follows"])
            biography = request.form["biography"]
            follow_ratio = followers / (follows + 1)
            bio_len = len(biography)
            bio_has_url = int(('http' in biography.lower()) or ('www' in biography.lower()))
            numeric_features = np.array([verified, follow_ratio, bio_len, bio_has_url], dtype=float)

            if "profile_img" not in request.files or request.files["profile_img"].filename == "":
                raise Exception("No image uploaded.")
            img_file = request.files["profile_img"]
            img = Image.open(img_file)
            img_feat = get_img_features(img)
            feat_vector = np.hstack([numeric_features, img_feat]).reshape(1, -1)
            feat_vector = scaler.transform(feat_vector)
            prediction = model.predict(feat_vector)[0]
            result = "fake" if prediction == 1 else "real"
            result_class = result
        except Exception as e:
            error = str(e)

    return render_template_string(HTML, result=result, result_class=result_class, error=error)

if __name__ == "__main__":
    app.run(debug=True)
