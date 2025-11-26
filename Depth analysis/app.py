from flask import Flask, render_template, request, redirect, session, g, url_for
import sqlite3, os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm  # progress bar

app = Flask(__name__)
app.secret_key = "crackdetect"

# ---------------- Database Object ----------------
DATABASE = "database.db"

# ---------------- Users Profile Pics Stored Directory ----------------
PROFILE_PIC = 'static/profile_pic'
app.config['PROFILE_PIC'] = PROFILE_PIC

# ---------------- Crack Check Upload Images Stored Directory ----------------
UPLOAD_CrackImages = 'static/CrackImages'
app.config['CrackImages'] = UPLOAD_CrackImages

# ---------------- Crack Check Result Stored Directory ----------------
UPLOAD_CrackResults = 'static/CrackResults'
app.config['CrackResults'] = UPLOAD_CrackResults

# ---------------- Index Page ----------------
@app.route("/")
def home():
    all_users=fetch_users()
    return render_template("index.html",all_users=all_users)

# ---------------- About Page ----------------
@app.route("/about")
def about():
    all_users=fetch_users()
    return render_template("about.html",all_users=all_users)

# ---------------- Services Page ----------------
@app.route("/services")
def services():
    all_users=fetch_users()
    return render_template("services.html",all_users=all_users)

# ---------------- Project Page ----------------
@app.route("/projects")
def projects():
    all_users=fetch_users()
    all_review=fetch_reviews()
    all_project=fetch_reviews()
    return render_template("projects.html",all_users=all_users)

# ---------------- Contact Page ----------------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# ---------------- Register ----------------
@app.route("/register",methods=['POST'])
def register():
    data = request.form
    os.makedirs(PROFILE_PIC, exist_ok=True)

    if 'img' not in request.files:
        return 'No file part'
    
    file = request.files['img']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['PROFILE_PIC'], unique_name)
        file.save(filepath)
        
    
    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO users (username, password, role, name, dob, email, contact, address, gender, img)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['uname'],
            data['pass'],
            3,
            data['fname'],
            data['dob'],
            data['email'],
            data['contact'],
            data['address'],
            data['gender'],
            unique_name,
            
        ))
    return '<script>alert("Registration Successful!"); window.location="/";</script>'

# ---------------- Login ----------------
@app.route("/login",methods=['POST'])
def login():
    username = request.form['uname']
    password = request.form['pass']
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = cursor.fetchone()

        if user:
            session['id']=user[0]
            session['uname']=user[1]
            session['role']=user[3]
            session['name']=user[4]
            session['dob']=user[5]
            session['email']=user[6]
            session['contact']=user[7]
            session['address']=user[8]
            session['gender']=user[9]
            session['img']=user[10]
            print(session['id'])
            return redirect(url_for('home'))
        else:
            return '<script>alert("Invalid username or password!"); window.location="/";</script>'

# ---------------- Logout ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))    

# ---------------- Crack Upload Page ----------------
@app.route("/crack")
def crack():
    return render_template("crack.html")

# ---------------- Crack Upload Page ----------------
@app.route("/crack_result", methods=['POST'])
def crack_result():

    os.makedirs(UPLOAD_CrackImages, exist_ok=True)

    if 'img' not in request.files:
        return 'No file part'
    
    file = request.files['img']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['CrackImages'], unique_name)
        file.save(filepath)

        prediction, result_image = predict_image(filepath)
        print("✅ Prediction:", prediction)

    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO prediction (userid, pred_result, pred_image, orginal_image, date_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session['id'],
            prediction,
            result_image,
            unique_name,
            datetime.now()
        ))

    return render_template("crack_result.html",prediction=prediction,result_image=result_image)

# ---------------- Register ----------------
@app.route("/add_engineers",methods=['POST'])
def add_engineers():
    data = request.form
    os.makedirs(PROFILE_PIC, exist_ok=True)

    if 'img' not in request.files:
        return 'No file part'
    
    file = request.files['img']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['PROFILE_PIC'], unique_name)
        file.save(filepath)
        
    
    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO users (username, password, role, name, dob, email, contact, address, gender, img)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['uname'],
            data['pass'],
            2,
            data['fname'],
            data['dob'],
            data['email'],
            data['contact'],
            data['address'],
            data['gender'],
            unique_name,
            
        ))
    return '<script>alert("New Engineer Add Successful!"); window.location="/";</script>'


# ---------------- Data Fetch Functions In Database Tables ----------------

# ---------------- Users ----------------
def fetch_users():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users')
        all_users = cursor.fetchall()
    return all_users

# ---------------- Projects ----------------
def fetch_projects():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM projects')
        all_projects = cursor.fetchall()
    return all_projects

# ---------------- Cracks ----------------
def fetch_cracks():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM crakscheck')
        all_crakscheck = cursor.fetchall()
    return all_crakscheck

# ---------------- Quotes ----------------
def fetch_quotes():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM quotes')
        all_quotes = cursor.fetchall()
    return all_quotes

# ---------------- Reviews ----------------
def fetch_reviews():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM reviews')
        all_reviews = cursor.fetchall()
    return all_reviews


# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ====================================
# 3. U-Net Model with Classification Head
# ====================================
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)

        self.bottom = self.contracting_block(256, 512)

        self.up3 = self.expansive_block(512, 256)
        self.up2 = self.expansive_block(256, 128)
        self.up1 = self.expansive_block(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # Classifier head (Normal vs Crack)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 2)   # 2 classes
        )

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        bottom = self.bottom(enc3)

        # classification head
        class_logits = self.classifier(bottom)

        dec3 = self.up3(bottom)
        dec2 = self.up2(dec3)
        dec1 = self.up1(dec2)

        mask = torch.sigmoid(self.final(dec1))  # segmentation output
        return mask, class_logits

# Reload model for testing
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("crack_unet.pth", map_location=DEVICE))
model.eval()

# Class labels
class_names = ["Normal", "Crack"]

def predict_image(img_path):
    """Run inference on a single image and show result."""
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_out, class_out = model(tensor)
        pred_class = class_out.argmax(dim=1).item()
        pred_label = class_names[pred_class]


    # Generate timestamp (example: 2025-10-17_09-45-22)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = f"{pred_label}_{timestamp}.png"
    result_path = os.path.join(app.config['CrackResults'], result_file)

    # show
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(f"Predicted: {pred_label}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(seg_out.squeeze().cpu(), cmap="gray")
    plt.title("Predicted Crack Mask")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(result_path, bbox_inches='tight', dpi=300)
    plt.close()

     


    return pred_label,result_file


# ---------------- RUN ----------------
if __name__ == "__main__":
        app.run(debug=True)
