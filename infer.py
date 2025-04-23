from util.pipnet import PIPNet, get_network
from util.args import get_args
from util.data import get_dataloaders
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

def load_model(device, args):
    # Prepare classes (you may not need the data loaders if you already know class count)
    _, _, _, _, _, _, _, classes = get_dataloaders(args, device)
    
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)

    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    device = 0
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        print("CPU used")
        device = torch.device('cpu')

    model = PIPNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer
    ).to(device)

    # model = nn.DataParallel(model, device_ids = device_ids)  

    model.eval()

    # Load checkpoint
    checkpoint = torch.load(args.state_dict_dir_net, map_location=device)

    # Fix: Remove "module." prefix if needed
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)

    return model, classes


def predict_single_image(image_path, model, device, args):
    # Apply same transform as training
    transform = get_transform(args)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[1]  # Output: logits
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class

def predict_with_prototypes(image_path, model, device, args, topk=5):
    # Transform image
    transform = get_inference_transform(args)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        proto_features, logits, min_distances = model(input_tensor)

        pred_class = torch.argmax(logits, dim=1).item()
        num_prototypes = model._classification.weight.shape[1]

        # Each prototype’s activation score
        proto_acts = torch.relu(1 - min_distances.squeeze(0))  # shape: [num_prototypes, H, W]
        proto_max_scores = proto_acts.flatten(start_dim=1).max(dim=1)[0]


        # Top-k activated prototypes
        topk_values, topk_idxs = torch.topk(proto_max_scores, topk)

        return {
            "predicted_class": pred_class,
            "topk_prototypes": [
                {"prototype_index": int(idx.item()), "score": float(score.item())}
                for idx, score in zip(topk_idxs, topk_values)
            ]
        }

def get_inference_transform(args):
    # These are always the same in your dataset functions
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform