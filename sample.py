import torch
from PIL import Image
import torchvision.transforms as T
from model import get_model
from data_loader import get_loader
import json

dataset, _ = get_loader()

model = get_model()

checkpoints = torch.load('checkpoints/model.pth')
model.load_state_dict(checkpoints['state_dict'])

#generate caption
def get_caps_from(features_tensors):
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors)
        caps,alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)

    idx2word = json.load(open('dataset/idx2word.json', 'r'))
    res = []
    for cap in caps:
        try:
            res.append(min(idx2word[cap], key= lambda x: len(x)))
        except:
            pass
    
    return res

transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

image = Image.open('image.jpg').convert('RGB')
image = transforms(image)
caps = get_caps_from(image.unsqueeze(0))

print(caps)
