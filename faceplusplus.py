import sys
import requests


def detect_text(path):
    r = requests.post('https://api-cn.faceplusplus.com/imagepp/v1/recognizetext', data={
        'api_key': '7zjTam74pqhjP96D0ZNcxjwO7w90TLgv',
        'api_secret': 'QQSvM1GDLfTRs05vWc7DAaudP671vb4B'
    }, files={
        'image_file': open(path, 'rb')
    })
    print(r.text)


if __name__ == '__main__':
    detect_text(sys.argv[1])
