# indentifysex

该模块通过判断照片中的人来判断其性别，男性返回1，女性返回0，可使用cpu
This module determines the gender of a person in a photograph and returns 1 for males and 0 for females with cpu
调用方法（Usage）：
from identifysex.idsex import identifysex
url = r'pic.jpg'
a = identifysex(url)
print(a)
