import face_recognition,requests,torch
from io import BytesIO
from .linearmodel import FastText
from .config import Config
import os
import torch.nn.functional as F

model = None

def load_model():
    global model
    model = FastText(config=Config())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(script_dir, './ptfiles/linearsex.pt')
    param = torch.load(pt_path, map_location=torch.device('cpu'))
    model.load_state_dict(param)
    model.eval()



def identifysex(url):
    """
    url 可以是文件，也可以是网络图片
    """
    load_model()
    try:
        """网络图片"""
        getimg = BytesIO(requests.get(url).content)
        img = face_recognition.load_image_file(getimg)
        fe = face_recognition.face_encodings(img)
    except:
        """现实图片"""
        img = face_recognition.load_image_file(url)
        fe = face_recognition.face_encodings(img)
    if fe != []:
        fe = [fe[0].tolist()]
        fe = torch.tensor([fe])
        res = F.softmax(model(fe),dim=1)
        result = torch.max(res.data,1).indices.numpy()[0]  # 女为[1,0],男为[0,1]
        return result
    else:
        return 'no face in picture'

if __name__ == '__main__':
    a = identifysex(r"http://www.miaoxingjundd.com/face/iJMf9DqM3264.jpg")
    print(a)
