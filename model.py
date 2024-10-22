import joblib
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from PIL import Image

# Initialize and load the model
class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(weights=None)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        return self.network(xb)

model_path = 'plantDisease-resnet34.pth'
base_model = Plant_Disease_Model()
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
# else:
#     print("No GPU available. Training will run on CPU.")
base_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
base_model.eval()

# Modify model to extract features
feature_extractor = nn.Sequential(*list(base_model.network.children())[:-1], nn.Flatten()) # only removed last layer
feature_extractor.eval()

# Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

svm_model = joblib.load('stacking_classifier.pkl')
def process_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def extract_features_single(image_path, feature_extractor):
    image = process_image(image_path, transform)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.numpy().flatten()

def predict(img_path):
    features = extract_features_single(img_path, feature_extractor)
    # print(type(features))

    data_features = features.reshape(1, -1)
    prediction = svm_model.predict(data_features)
    print(f"Predicted class for the new sample: {prediction[0]}")
    return prediction[0]


