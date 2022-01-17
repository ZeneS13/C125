from flask import Flask, jsonify,request
from model import getPred

app=Flask(__name__)

@app.route("/predDigit",methods=["POST"])
def predData():
    image=request.files.get("digitImg")
    predict=getPred(image)

    return jsonify({
        "Prediction":predict,
        "message":"this is the value predicted"
    })


if __name__=="__main__":
    app.run(debug=True)