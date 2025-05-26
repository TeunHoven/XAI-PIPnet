# XAI-PIPnet

This project was created as part of the course **Interactive and Explainable AI Design** at JADS.

We are **PIP-Net-1**, and this project explores how visual explanations impact user understanding and trust in AI-based bird species classification systems.

## Project Objective

This project investigates the question:

> **Do users prefer seeing only local explanations or both local and global explanations when interpreting AI decisions?**

To answer this, we built **PeepNet** — a web-based interface that:

- Predicts bird species using an interpretable deep learning model: **PIP-Net**.
- Visualizes **local prototypes** found in the uploaded image.
- Optionally displays **global prototypes** that were learned but **not activated** during the prediction.

## Project Structure

```bash
XAI-PIPNET/
├── app.py                       # Flask web app
├── infer.py                     # Model loading and prediction logic
├── util/                        # Helper scripts (args, visualization, etc.)
├── pipnet/                      # Model checkpoints and visualised prototype folders
│   ├── visualised_prototypes/   # Necessary to visualize the learned prototypes
│   ├── checkpoints/
│   └── out.txt                  # Prototype-to-class mapping
├── data/
│   └── CUB_200_2011/            # Necessary for the code to load the dataloader
├── static/                      # Saved image outputs for frontend
│   ├── uploads/
│   ├── local_prototypes/
│   ├── global_prototypes/
│   └── css/
├── templates/                   # HTML templates for Flask
├── runs/                        # Model inference output folder
├── requirements.txt
├── README.md
└── .gitignore
```

> ⚠️ **Important:**  
> - Ensure that `pipnet/visualised_prototypes/` and `data/CUB_200_2011/` exist and are correctly populated.  
> - Without these folders, **the application will not run properly.**


## Usage Instructions

1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.8+ and a CUDA-compatible GPU (optional for faster inference).

2. Load Pretrained Model

    Ensure the model checkpoint exists at:

    - `pipnet/checkpoints/pipnet_cub_trained`

You can download a pretrained checkpoint from the original PIP-Net repository or train your own.

3. Run the Web App

    `python app.py`

The app runs locally at http://127.0.0.1:5000

4. Upload an Image

    Use the web interface to upload a bird image:

    - */*: View the final prototype (demo).

    - */local*: View only local explanations.

    - */global*: View only global explanations.