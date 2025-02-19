import face_recognition,requests,torch
from io import BytesIO
from linearmodel import FastText
from config import Config
import os
import torch.nn.functional as F


model = None

def load_model():
    global model
    model = FastText(config=Config())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(script_dir, './ptfiles/linearsex-32-norm-20250218.pt')
    param = torch.load(pt_path, map_location=torch.device('cpu'))
    model.load_state_dict(param)
    model.eval()


def identifysex(url,num=1):
    """
    url 可以是文件，也可以是网络图片
    """
    load_model()
    try:
        """网络图片"""
        response = requests.get(url)
        if response.status_code == 200:
            getimg = BytesIO(response.content)
            img = face_recognition.load_image_file(getimg)
            loc = face_recognition.face_locations(img, number_of_times_to_upsample=1)
            fe = face_recognition.face_encodings(img, loc)
        else:
            return '请重新填写url地址'
    except:
        """现实图片"""
        img = face_recognition.load_image_file(url)
        loc = face_recognition.face_locations(img, number_of_times_to_upsample=0)
        fe = face_recognition.face_encodings(img, loc)

    if fe != [] and loc != []:
        fe = torch.tensor([[fe[num - 1]]], dtype=torch.float32)
        modelres = model(fe)
        print(f'这个时候是导入模型之后，模型对象的预测结果是{modelres}')
        res = F.softmax(modelres, dim=1)
        print(f'这个是在模型预测结果后，通过softmax方法取出dim维度的值（这里只有一维，所以dim是1），这个值是{res}，我们用到其中的res.data:{res.data}')
        print(f'torch.max是去在所有概率中最大的那个概率，indices是最大概率的索引')
        result = torch.max(res.data, 1).indices.numpy()[0]
        print(f'这个是取最大概率的值的展示：{torch.max(res.data, 1)}，其中value为概率，indices为索引')
        print(f'这个返回了最大的索引，这个1代表了返回的索引个数。结果为：{result}，0为女，1为男')
        return result

    else:
        return 'no face in picture'

if __name__ == '__main__':
    a = identifysex(r"http://www.miaoxingjundd.com/face/Ba2jjAF09327.jpg")
    print(a)
