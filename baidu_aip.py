import sys
from aip import AipImageClassify
from aip import AipOcr


APP_ID = '10560205'
API_KEY = 'HvEK74yZQTrZ2DROK8pubvCg'
SECRET_KEY = '7ZGHp9pkG1g8KfBZsuI1pGOUVf5KKpiL'


def detect_logos(path):
    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
    f = open(path, 'rb')
    image = f.read()
    f.close()
    print(client.logoSearch(image, {
        'custom_lib': 'true'
    }))


def detect_texts(path):
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    f = open(path, 'rb')
    image = f.read()
    f.close()
    print(client.basicGeneral(image))


if __name__ == '__main__':
    detect_logos(sys.argv[1])
    detect_texts(sys.argv[1])
