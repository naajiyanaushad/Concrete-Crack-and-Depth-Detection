from flask import Flask, json, render_template, request, redirect, session, g, url_for
import sqlite3, os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
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

# ---------------- Users Profile Pics Stored Directory ----------------
PROJECT_PIC = 'static/project_pic'
app.config['PROJECT_PIC'] = PROJECT_PIC

# ---------------- Crack Check Upload Images Stored Directory ----------------
UPLOAD_CrackImages = 'static/CrackImages'
app.config['CrackImages'] = UPLOAD_CrackImages

# ---------------- Crack Check Result Stored Directory ----------------
UPLOAD_CrackResults = 'static/CrackResults'
app.config['CrackResults'] = UPLOAD_CrackResults

#Convert NumPy → Python types BEFORE json.dumps
def to_python_float_list(arr):
    if arr is None:
        return None
    return [float(x) for x in arr]

def to_python_dict(d):
    if d is None:
        return None
    return {k: str(v) for k, v in d.items()}

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
    all_project=fetch_projects()
    return render_template("projects.html",all_users=all_users,all_project=all_project)

# ---------------- Project Page ----------------
@app.route("/projectdetail/<int:pid>")
def projectdetail(pid):
    all_users=fetch_users()
    projects=fetch_sproject(pid)
    return render_template("project-details.html",all_users=all_users,projects=projects)

# ---------------- Contact Page ----------------
@app.route("/contact")
def contact():
    all_users=fetch_users()
    return render_template("contact.html",all_users=all_users)

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
    all_users=fetch_users()
    return render_template("crack.html",all_users=all_users)

# ---------------- Crack Upload Page ----------------
@app.route("/crack_result", methods=['POST'])
def crack_result():
    all_users=fetch_users()

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

        label, avg_depths, confidence, images = predict_image(filepath)
        print("✅ Prediction:", label, avg_depths, confidence, images)

    # Convert images dict to JSON (SQLite safe)
    images_json = json.dumps(to_python_dict(images))
    avg_depths_json = json.dumps(to_python_float_list(avg_depths))
    confidence = float(confidence)

    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO prediction (userid, pred_result, pred_image, orginal_image, avg_depths, confidence, date_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['id'],
            label,
            images_json,
            unique_name,
            avg_depths_json,
            confidence,
            datetime.now()
        ))

    return render_template(
        "crack_result.html",
        label=label,
        confidence=f"{confidence*100:.2f}",
        avg_depths=avg_depths,
        images=images,
        unique_name = unique_name,
        cracked=(label == "CRACKED"),
        all_users=all_users
        )

# ---------------- Add Engineers----------------
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

# ---------------- Add Project----------------
@app.route("/add_project",methods=['POST'])
def add_project():
    data = request.form
    os.makedirs(PROJECT_PIC, exist_ok=True)

    if 'img' not in request.files:
        return 'No file part'
    
    file = request.files['img']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['PROJECT_PIC'], unique_name)
        file.save(filepath)
        
    
    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO projects (pcat, pname, ptype, duration, area, client, location, bheight, ftype, fmaterial, pspace, erate, sdesc, poverview, status, img)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['pcat'],
            data['pname'],
            data['ptype'],
            data['duration'],
            data['area'],
            data['client'],
            data['loc'],
            data['bheight'],
            data['ftype'],
            data['fmaterial'],
            data['pspace'],
            data['erate'],
            data['sdes'],
            data['poverview'],
            data['pstatus'],
            unique_name

        ))
    return '<script>alert("Project Added Successful!"); window.location="/projects";</script>'

# ---------------- Profile Page ----------------
@app.route("/profile")
def profile():
    all_users=fetch_users()
    profile_data=fetch_profile()
    print("prodata: ",profile_data)
    return render_template("profile.html",profile_data=profile_data,all_users=all_users)

# ---------------- Update Profile ----------------
@app.route("/update_profile",methods=['POST'])
def update_profile():
    data = request.form
    
    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE users SET username=?, name=?, dob=?, email=?, contact=?, address=?, gender=? WHERE id = ?
        ''', (
            data['uname'],
            data['fname'],
            data['dob'],
            data['email'],
            data['contact'],
            data['address'],
            data['gender'],
            session['id']
        ))
    return '<script>alert("Profile Updated Successful!"); window.location="/profile";</script>'

# ---------------- Update Password ----------------
@app.route("/update_password",methods=['POST'])
def update_password():
    data = request.form
    
    with sqlite3.connect(DATABASE) as conn:
     
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE users SET password=? WHERE id = ?
        ''', (
            data['pass'],
            session['id']
        ))
    return '<script>alert("Password Updated Successful!"); window.location="/profile";</script>'

# ---------------- Update Profile Pic ----------------
@app.route("/updatepropic",methods=['POST'])
def updatepropic():
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
            UPDATE users SET img=? WHERE id = ?
        ''', (
            unique_name,
            session['id']
        ))
    return '<script>alert("Profile Pic Updated Successful!"); window.location="/profile";</script>'



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

# ---------------- Single Projects ----------------
def fetch_sproject(pid):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM projects WHERE id=?',(pid,))
        project = cursor.fetchone()
    return project

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

# ---------------- Reviews ----------------
def fetch_profile():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id=?',(session['id'],))
        profile_data = cursor.fetchall()
    return profile_data

# ====================================
# 3. U-Net Model with Classification Head
# ====================================
class CrackClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16, 2)
        )

    def forward(self, x):
        return self.net(x)


class FastUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))
        d1 = torch.nn.functional.interpolate(e2, scale_factor=2)
        d2 = torch.relu(self.dec1(d1))
        return torch.sigmoid(self.out(d2))
    
# =====================================================
# Load models
# =====================================================
classifier = CrackClassifier()
classifier.load_state_dict(torch.load("sdnet_classifier.pth", map_location="cpu"))
classifier.eval()

depth_model = FastUNet()
depth_model.load_state_dict(torch.load("sdnet_fast_depth_model.pth", map_location="cpu"))
depth_model.eval()

# =====================================================
# Preprocessing
# =====================================================
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def future_depth(depth, years, alpha=0.06):
    return np.clip(depth * (1 + alpha*years), 0, 1)


def predict_image(img_path):
    """Run inference on a single image and show result."""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    unique_id = uuid.uuid4().hex 

    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(1).item()
        confidence = probs.max().item()

    label = "CRACKED" if pred_class == 1 else "UNCRACKED"

    image_paths = {}
    depths = {}
    avg_depths=[]
    if label == "CRACKED":
        with torch.no_grad():
            depth_now = depth_model(x)[0,0].numpy()

        depths = {
            "now": depth_now,
            "1": future_depth(depth_now, 1),
            "5": future_depth(depth_now, 5),
            "10": future_depth(depth_now, 10)
        }

            # Save depth images
        for k,v in depths.items():
            path = f"{UPLOAD_CrackResults}/depth_{k}_{unique_id}.png"
            plt.imsave(path, v, cmap="inferno")
            image_paths[f"depth_{k}"] = path

             # Depth growth graph
        avg_depths = [
            depth_now.mean(),
            depths["1"].mean(),
            depths["5"].mean(),
            depths["10"].mean()
        ]

        plt.figure()
        plt.plot([0,1,5,10], avg_depths, marker="o")
        plt.xlabel("Years")
        plt.ylabel("Average Depth")
        plt.title("Crack Depth Growth")
        plt.grid(True)
        graph_path = f"{UPLOAD_CrackResults}/growth_{unique_id}.png"
        plt.savefig(graph_path)
        plt.close()

        image_paths["growth"] = graph_path

    return label,avg_depths,confidence, image_paths


# ---------------- RUN ----------------
if __name__ == "__main__":
        app.run(debug=True)
