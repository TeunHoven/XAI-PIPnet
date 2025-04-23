from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
from infer import predict_with_prototypes
from util.args import get_args
from util.visualize_prediction import vis_pred, vis_pred_experiments
from infer import load_model
import torch
import shutil

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args()
args.state_dict_dir_net = "pipnet/checkpoints/pipnet_cub_trained" 
args.epochs = 0
args.epochs_pretrain = 0
args.dir_for_saving_images = "static/saved_images"
args.wshape = 26 # DO NOT CHANGE THIS
model, classes = load_model(device, args)

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
        name = " ".join(name.split('_')).title()

        entries.append((score, name, folder))

    return entries, total_score


def predict_bird(filename):
    vis_pred_experiments(model, app.config['UPLOAD_FOLDER'], classes, device, args)

    entries, total_score = get_score(filename.split('.')[0])

    # Get the entry with the highest number
    if entries:
        highest = max(entries, key=lambda x: x[0])
        print("Highest entry:", highest)
    else:
        print("No valid folders found.")

    class_name = highest[1]
    percentage = highest[0] / total_score * 100
    folder = highest[2]

    return class_name, percentage, folder

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

    print("Prototypes copied to static/local_prototypes/explanations folder.")
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

            print("Class name:", class_name)

            explanations = get_explanations_prototypes(prototype_id, class_name)

            prototypes.append({
                "id": prototype_id,
                "similarity": float(similarity),
                "weight": float(weight),
                "explanations": explanations
            })

    # sort prototypes by similarity
    prototypes.sort(key=lambda x: float(x["similarity"]), reverse=True)

    print("Prototypes copied to static/local_prototypes folder.")

    return prototypes

def get_global_prototypes(class_name, local_prototypes=None):
    local_prototypes = [prototype["id"] for prototype in local_prototypes]

    class_name_in_file = class_name.replace(" ", "_")
    parent_folder = app.config["PIPNET_FOLDER"]

    folders = [name for name in os.listdir(parent_folder)
           if os.path.isdir(os.path.join(parent_folder, name))]
    
    global_prototypes = []
    
    for folder in folders:
        highest_similarity = 0
        best_prototype_file = None
        best_prototype_file_path = None

        for filename in os.listdir(os.path.join(parent_folder, folder)):
            prototype_id = filename.split('_')[0].split('p')[-1]

            if prototype_id in local_prototypes:
                break

            if class_name_in_file in filename:
                prototype = os.path.join(parent_folder, folder, filename)

                if os.path.isfile(prototype):
                    similarity = float(filename.split('_')[2])

                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_prototype_file = filename
                        best_prototype_file_path = prototype

        if best_prototype_file is not None:
            dst_file = os.path.join("static/global_prototypes", best_prototype_file)
            shutil.copy2(best_prototype_file_path, dst_file)

            prototype_id = best_prototype_file.split('_')[0].split('p')[-1]
            similarity = filename.split('_')[2].split('sim')[-1]

            global_prototypes.append({
                "id": prototype_id,
                "similarity": float(similarity)
            })
        
        best_prototype_file = None
        best_prototype_file_path = None

    return global_prototypes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "images", filename)
            file.save(filepath)

            class_name, percentage, folder = predict_bird(filename)

            local_prototypes = get_local_prototypes(filename.split(".")[0], folder, class_name)
            
            for local_prototype in local_prototypes:
                if len(local_prototype["explanations"]) > 3:
                    # Get 3 random explanations
                    local_prototype["explanations"] = np.random.choice(local_prototype["explanations"], size=3, replace=False).tolist()

            global_prototypes = get_global_prototypes(class_name, local_prototypes)

            return render_template('index.html', filename=filename, prediction=class_name, percentage=percentage, local_prototypes=local_prototypes, global_prototypes=global_prototypes)
        
    return render_template('index.html', filename=None)

@app.route('/display/prototypes/global/<prototype_id>')
def display_global_prototype(prototype_id):
    file = None
    for filename in os.listdir('static/global_prototypes'):
        if os.path.isfile(os.path.join('static/global_prototypes', filename)):
            if prototype_id == filename.split('_')[0].split('p')[-1]:
                file = filename
                break

    return redirect(url_for('static', filename='global_prototypes/' + file), code=301)

@app.route('/display/prototypes/local/<prototype_id>')
def display_local_prototype(prototype_id):
    file = None
    for filename in os.listdir('static/local_prototypes'):
        if os.path.isfile(os.path.join('static/local_prototypes', filename)):
            if prototype_id == filename.split('_')[1].split('p')[-1]:
                file = filename
                break

    return redirect(url_for('static', filename='local_prototypes/' + file), code=301)

@app.route('/display/prototypes/local/explanations/<prototype_id>')
def display_local_explanations(prototype_id):
    file = None
    for filename in os.listdir('static/local_prototypes/explanations/'):
        if os.path.isfile(os.path.join('static/local_prototypes/explanations', filename)):
            if prototype_id == filename.split('_')[1].split('p')[-1]:
                file = filename
                break

    return redirect(url_for('static', filename='local_prototypes/explanations/' + file), code=301)

@app.route('/display/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/images/' + filename), code=301)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
