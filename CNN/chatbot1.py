# start

from flask import Flask, request, Response
from fu_lib1 import *


app = Flask(__name__)


@app.route('/faceai', methods=['POST'])
def faceai():
    """
    For dealing with the POST request.
    :return:
    """

    data = request.files
    image1 = data.get('my_animal').read()
    with open('my_animal.jpg', "wb") as f:
        f.write(image1)
    image2 = data.get('target').read()
    with open('target.jpg', "wb") as f:
        f.write(image2)

    reply()
    with open("final.jpg", "rb") as f:
        return_data = f.read()

    return Response(return_data,  content_type="image/jpeg")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
