from model import respond, send_message
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(respond(userText))

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)