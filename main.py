import io
from flask import Flask, jsonify, request
import torch
from PIL import Image
import torchvision.transforms as T
from model import get_model
from data_loader import get_loader
import json
import base64
import logging

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods = ['POST'])
def predict():
    dataset, _ = get_loader()

    model = get_model()

    checkpoints = torch.load('checkpoints/model.pth')
    model.load_state_dict(checkpoints['state_dict'])

    def softIOU(out, target):
        i = len(set(out).intersection(set(target)))
        u = len(set(out).union(set(target)))
        return i / u

    #generate caption
    def get_caps_from(features_tensors):
        model.eval()
        with torch.no_grad():
            features = model.encoder(features_tensors)
            caps, _ = model.decoder.generate_caption(features,vocab=dataset.vocab)
        if '<EOS>' in caps:
            caps.remove('<EOS>')

        layer1 = json.load(open('dataset/layer1.json', 'r'))
        layer2 = json.load(open('dataset/layer2.json', 'r'))

        res_id = 0
        res_ingrs = []
        for recipe in layer2:
            id = recipe['id']
            ingrs = recipe['ingredients']
            if softIOU(ingrs, caps) > softIOU(res_ingrs, caps):
                res_ingrs = ingrs
                res_id = id

        title = layer1[res_id]['title']

        idx2word = json.load(open('dataset/idx2word.json', 'r'))
        res = []
        for cap in caps:
            try:
                res.append(min(idx2word[cap], key= lambda x: len(x)))
            except:
                pass
        if '' in res:
            res.remove('') 
        return {'title': title, 'ingredients': list(set(res))} 

    transforms = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    
    app.logger.info(json.loads(request.data).keys())

    image_bytes = json.loads(request.data)['image']
    image_data = base64.b64decode(image_bytes)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = transforms(image)
    result = get_caps_from(image.unsqueeze(0))

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)