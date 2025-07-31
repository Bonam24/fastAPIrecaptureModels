import torch
import timm
from PIL import Image
from torchvision import transforms

def load_all_models():
    models = {
        "moire": {
            "path": "models/large_new_moire_detection_model_with_metadata.pt",
            "threshold": 0.2
        },
        "computer": {
            "path": "models/large_computer_screen_detection_model_with_metadata.pt",
            "threshold": 0.5
        },
        "phone": {
            "path": "models/phone_screen_detection_model_with_metadata.pt",
            "threshold": 0.5
        },
        "printed": {
            "path": "models/printed_sources_detection_model_with_metadata.pt",
            "threshold": 0.5
        },
        "tv": {
            "path": "models/tv_screen_detection_model_with_metadata.pt",
            "threshold": 0.5
        }
    }

    model_info = {}

    for name, config in models.items():
        try:
            checkpoint = torch.load(config["path"], map_location=torch.device('cpu'))
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            class_names = checkpoint.get('class_names', ['recaptured', 'real'])

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            model_info[name] = {
                "model": model,
                "transform": transform,
                "class_names": class_names,
                "threshold": config["threshold"]
            }

        except Exception as e:
            print(f"Error loading {name} model: {e}")
            continue

    return model_info


def predict_image(image, model_info):
    """image is a PIL image"""
    img = image.convert('RGB')
    predictions = {}
    final_verdict = "camera"

    for name, info in model_info.items():
        try:
            img_tensor = info["transform"](img).unsqueeze(0)

            with torch.no_grad():
                output = info["model"](img_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze()

                recaptured_idx = info["class_names"].index('recaptured')
                real_idx = info["class_names"].index('real')

                recaptured_conf = probabilities[recaptured_idx].item()
                real_conf = probabilities[real_idx].item()

                if recaptured_conf >= info["threshold"]:
                    model_verdict = "recaptured"
                    final_verdict = "screen"
                else:
                    model_verdict = "real"

                predictions[name] = {
                    "class": model_verdict,
                    "recaptured_conf": f"{recaptured_conf:.6f}",
                    "real_conf": f"{real_conf:.6f}",
                    "threshold": info["threshold"],
                    "model_name": name.replace("_", " ").title()
                }

        except Exception as e:
            predictions[name] = {"error": str(e)}

    return {
        "model_predictions": predictions,
        "final_prediction": final_verdict,
        "moire_threshold": model_info.get("moire", {}).get("threshold", 0.2)
    }






