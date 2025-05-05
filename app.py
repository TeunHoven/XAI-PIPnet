from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
from infer import predict_with_prototypes
from util.args import get_args
from util.visualize_prediction import vis_pred, vis_pred_experiments
from infer import load_model
import time
import torch
import shutil

import numpy as np

import re
import ast
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args()
args.state_dict_dir_net = "pipnet/checkpoints/pipnet_cub_trained" 
args.epochs = 0
args.epochs_pretrain = 0
args.dir_for_saving_images = "static/saved_images"
args.wshape = 26 # DO NOT CHANGE THIS, IDK WHY BUT IT WORKS WITH THIS VALUE
model, classes = load_model(device, args)

# ========== LOAD RELEVANT PROTOTYPES ==========
# Load the file
with open("pipnet/out.txt", "r") as f:
    content = f.read()

# Regex pattern to extract class and prototypes
pattern = re.compile(r"Class (\d+) .*?: has \d+ relevant prototypes:  (\[.*?\])")

# Dictionary to store results
relevant_protypes = defaultdict(list)

# Extract matches and populate dictionary
for match in pattern.finditer(content):
    class_index = int(match.group(1))
    prototypes = ast.literal_eval(match.group(2))
    relevant_protypes[class_index] = prototypes

print("Relevant prototypes loaded.")
print("Relevant prototypes:", relevant_protypes[17])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RUN_FOLDER'] = 'runs/run_pipnet/static/saved_images/Experiments'
app.config['PIPNET_FOLDER'] = 'pipnet/visualised_prototypes'

def get_score(filename):
    parent_folder = os.path.join(app.config["RUN_FOLDER"], filename)
    # List all subdirectories in the parent folder
    folders = [name for name in os.listdir(parent_folder)
           if os.path.isdir(os.path.join(parent_folder, name))]

    entries = []

    total_score = 0

    for folder in folders:
        score = float(folder.split('_')[0])
        total_score += score

        name = folder.split('.')[-1]
        id = folder.split('_')[1].split('.')[0]
        name = " ".join(name.split('_')).title()

        entries.append((id, score, name, folder))

    return entries, total_score


def predict_bird(filename):
    vis_pred_experiments(model, app.config['UPLOAD_FOLDER'], classes, device, args)

    entries, total_score = get_score(filename.split('.')[0])

    # Get the entry with the highest number
    if entries:
        highest = max(entries, key=lambda x: x[1])
        print("Highest entry:", highest)
    else:
        print("No valid folders found.")

    class_id = highest[0]
    class_name = highest[2]
    score = highest[1]
    folder = highest[3]

    if score < 5.0:
        print("No valid class found.")
        return None, None, None, None

    return class_id, class_name, score, folder

def get_explanations_prototypes(prototype_id, class_name):
    class_name = class_name.replace(" ", "_")
    folder_path = app.config['PIPNET_FOLDER']

    explanations = []

    for foldername in os.listdir(folder_path):
        if prototype_id == foldername.split('_')[1]:
            for filename in os.listdir(os.path.join(folder_path, foldername)):
                if class_name in filename:
                    src_file = os.path.join(folder_path, foldername, filename)
                    dst_file = os.path.join("static/local_prototypes/explanations", filename)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                    
                    explanations.append(filename)

    return explanations

def get_local_prototypes(folder_name, class_folder, class_name):
    folder_path = os.path.join(app.config['RUN_FOLDER'], folder_name, class_folder)

    prototypes = []
    
    # Loop through and copy only "rect" files
    for filename in os.listdir(folder_path):
        if "rect" in filename:
            src_file = os.path.join(folder_path, filename)
            dst_file = os.path.join("static/local_prototypes", filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)

            # Get number from filename
            # Assuming the filename format is like "mul2.418_p758_sim0.833_w2.904_rect.png" and we need the number 758 from p758
            prototype_id = filename.split('_')[1].split('p')[-1]
            similarity = filename.split('_')[2].split('sim')[-1]
            weight = filename.split('_')[3].split('w')[-1]

            explanations = get_explanations_prototypes(prototype_id, class_name)

            prototypes.append({
                "id": prototype_id,
                "similarity": float(similarity),
                "weight": float(weight),
                "explanations": explanations
            })

    # sort prototypes by similarity
    prototypes.sort(key=lambda x: float(x["similarity"] * x["weight"]), reverse=True)

    print("Prototypes copied to static/local_prototypes folder.")

    return prototypes

def get_global_prototypes(class_name, local_prototypes=None):
    class_name_in_file = class_name.replace(" ", "_")
    parent_folder = app.config["PIPNET_FOLDER"]

    folders = [name for name in os.listdir(parent_folder)
           if os.path.isdir(os.path.join(parent_folder, name))]
    
    global_prototypes = []
    
    for folder in folders:
        all_images = {}

        prototype_id = folder.split('_')[1]

        if int(prototype_id) not in local_prototypes:
            continue

        for filename in os.listdir(os.path.join(parent_folder, folder)):
            if class_name_in_file in filename:
                prototype = os.path.join(parent_folder, folder, filename)

                if os.path.isfile(prototype):
                    all_images[filename] = prototype

        all_files = list(all_images.keys())
        chosen_file = np.random.choice(all_files, size=1, replace=False).tolist()[0]
        dst_file = os.path.join("static/global_prototypes", chosen_file)
        chosen_file_path = all_images[chosen_file]
        shutil.copy2(chosen_file_path, dst_file)

        prototype_id = chosen_file.split('_')[0].split('p')[-1]
        similarity = chosen_file.split('_')[2].split('sim')[-1]

        global_prototypes.append({
            "id": int(prototype_id),
            "similarity": float(similarity)
        })

    # sort prototypes by id
    global_prototypes.sort(key=lambda x: x["id"])

    print(global_prototypes)

    return global_prototypes

def remove_all_files():
    # Remove all files in the uploads folder
    for filename in os.listdir("static/uploads/images"):
        file_path = os.path.join("static/uploads/images", filename)
        if os.path.isfile(file_path): 
            os.remove(file_path)

    # Remove all files in the local_prototypes folder
    for filename in os.listdir('static/local_prototypes'):
        file_path = os.path.join('static/local_prototypes', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Remove all files in the local_prototypes/explanations folder
    for filename in os.listdir('static/local_prototypes/explanations'):
        file_path = os.path.join('static/local_prototypes/explanations', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Remove all files in the global_prototypes folder
    for filename in os.listdir('static/global_prototypes'):
        file_path = os.path.join('static/global_prototypes', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    return True

def get_relevant_prototypes(class_name, local_prototypes):
    relevant_prototypes = []

    for prototype in local_prototypes:
        if class_name in prototype["explanations"]:
            relevant_prototypes.append(prototype)

    return relevant_prototypes

@app.route('/local', methods=['GET', 'POST'])
def index_local():
    if request.method == 'POST':

        if remove_all_files():
            print("All files removed successfully.")
        else:
            print("Failed to remove files.")

        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "images", filename)
            file.save(filepath)

            class_id, class_name, score, folder = predict_bird(filename)

            if class_id is None:
                return render_template('index.html', filename=filename, prediction=None, time=time)
            
            # only get the prototype id from the filename
            class_relevant_prototypes = relevant_protypes[int(class_id)-1]
            class_relevant_prototypes = [p[0] for p in class_relevant_prototypes]
            print(class_relevant_prototypes)

            print("Relevant prototypes:", class_relevant_prototypes)

            local_prototypes = get_local_prototypes(filename.split(".")[0], folder, class_name)
            
            for local_prototype in local_prototypes:
                if len(local_prototype["explanations"]) > 3:
                    # Get 3 random explanations
                    local_prototype["explanations"] = np.random.choice(local_prototype["explanations"], size=3, replace=False).tolist()

            # difference between local_prototypes and relevant_prototypes
            global_prototypes = [relevant_prototype for relevant_prototype in class_relevant_prototypes if relevant_prototype not in [int(prototype_id["id"]) for prototype_id in local_prototypes]]

            print("Global prototypes:", global_prototypes)

            if len(global_prototypes) == 0:
                global_prototypes = None
            else:
                global_prototypes = get_global_prototypes(class_name, global_prototypes)

            return render_template('index_only_local.html', filename=filename, prediction=class_name, score=score, local_prototypes=local_prototypes, global_prototypes=global_prototypes, time=time)
        
    return render_template('index_only_local.html', filename=None, time=time)

@app.route('/global', methods=['GET', 'POST'])
def index_global():
    if request.method == 'POST':

        if remove_all_files():
            print("All files removed successfully.")
        else:
            print("Failed to remove files.")

        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "images", filename)
            file.save(filepath)

            class_id, class_name, score, folder = predict_bird(filename)

            if class_id is None:
                return render_template('index.html', filename=filename, prediction=None, time=time)
            
            # only get the prototype id from the filename
            class_relevant_prototypes = relevant_protypes[int(class_id)-1]
            class_relevant_prototypes = [p[0] for p in class_relevant_prototypes]
            print(class_relevant_prototypes)

            print("Relevant prototypes:", class_relevant_prototypes)

            local_prototypes = get_local_prototypes(filename.split(".")[0], folder, class_name)
            
            for local_prototype in local_prototypes:
                if len(local_prototype["explanations"]) > 3:
                    # Get 3 random explanations
                    local_prototype["explanations"] = np.random.choice(local_prototype["explanations"], size=3, replace=False).tolist()

            # difference between local_prototypes and relevant_prototypes
            global_prototypes = [relevant_prototype for relevant_prototype in class_relevant_prototypes if relevant_prototype not in [int(prototype_id["id"]) for prototype_id in local_prototypes]]

            print("Global prototypes:", global_prototypes)

            if len(global_prototypes) == 0:
                global_prototypes = None
            else:
                global_prototypes = get_global_prototypes(class_name, global_prototypes)

            return render_template('index.html', filename=filename, prediction=class_name, score=score, local_prototypes=local_prototypes, global_prototypes=global_prototypes, time=time)
        
    return render_template('index.html', filename=None, time=time)

@app.route('/display/prototypes/global/<prototype_id>?time=<time>')
def display_global_prototype(prototype_id, time):
    folder = 'static/global_prototypes'
    for filename in os.listdir(folder):
        if filename.startswith(f'p{prototype_id}_') and os.path.isfile(os.path.join(folder, filename)):
            print("Found match:", filename)
            return redirect(url_for('static', filename=f'global_prototypes/{filename}'))
    print("No match found for prototype ID:", prototype_id)

@app.route('/display/prototypes/local/<prototype_id>?time=<time>')
def display_local_prototype(prototype_id, time):
    file = None
    for filename in os.listdir('static/local_prototypes'):
        if os.path.isfile(os.path.join('static/local_prototypes', filename)):
            if prototype_id == filename.split('_')[1].split('p')[-1]:
                file = filename
                break

    return redirect(url_for('static', filename='local_prototypes/' + file), code=301)

@app.route('/display/prototypes/local/explanations/<prototype_id>?time=<time>')
def display_local_explanations(prototype_id, time):
    file = None
    for filename in os.listdir('static/local_prototypes/explanations/'):
        if os.path.isfile(os.path.join('static/local_prototypes/explanations', filename)):
            if prototype_id == filename.split('_')[1].split('p')[-1]:
                file = filename
                break

    return redirect(url_for('static', filename='local_prototypes/explanations/' + file), code=301)

@app.route('/display/uploads/<filename>?time=<time>')
def display_image(filename, time):
    return redirect(url_for('static', filename='uploads/images/' + filename), code=301)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
