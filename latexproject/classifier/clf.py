import torch
import torch.nn as nn
from torchvision import models, transforms
import PIL

weight_path = './best_model.pt'

class resnet(nn.Module):
    def __init__(self, class_n=2, rate = 0.1):
        super(resnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.classification_layer = nn.Linear(1000,2)
        self.dropout = nn.Dropout(rate)

    def forward(self, inputs):
        output = self.model(inputs)
        output = self.dropout(self.classification_layer(output))

        return output

def load_clf_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet().to(device)
    model.load_state_dict(torch.load(weight_path,map_location=device))
    return model


def do_predict(model, img_path):
    ##이미지 변환
    test_transforms = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

    img = PIL.Image.open(img_path)
    trans_img = test_transforms(img).unsqueeze(0)
    model.eval()
    v, pred = torch.max(model(trans_img).data,1)

    return v, pred

