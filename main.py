# encoding: utf-8
# time    : 2021-06-06 14:19
# author  : 丛星星
"""
用于前后端交互。
"""
import uuid
from flask import Flask, render_template, request, send_from_directory
import numpy as np
from ClassificationModel import ClassificationModel

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("main.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    return render_template("main.html")

# 单条预测
@app.route("/get_text_to_predict", methods=["GET", 'POST'])
def get_text_to_predict():
    text_title = request.values['text_title']
    text_content = request.values['text_content']
    label_name, score = model.predict_with_title(content=text_content, title=text_title)
    if score[0][np.argmax(score)] < 0.8:
        label_name = "其他"
    return label_name

# 批量预测
@app.route("/get_file_to_predict", methods=['POST'])
def get_file_to_predict():
    news_file = request.files.get("news_file")
    file_dir = "static/upload/"
    file_name = str(uuid.uuid4())+'.xlsx'
    file_path = file_dir+file_name
    # 保存文件
    news_file.save(file_path)
    model.predict_label_write_to_file(file_path)
    # 下载文件
    return send_from_directory(file_dir, file_name, as_attachment=True)


# 下载上传文件格式模板
@app.route("/get_file_template", methods=["GET", 'POST'])
def get_file_template():
    file_dir = "static/file/"
    file_name = "upload_temple.xlsx"
    # 下载文件模板
    return send_from_directory(file_dir, file_name, as_attachment=True)


model = ClassificationModel()
# 初始化加载模型
def init_model():
    global model
    vocab_file = "static/model/vocab.txt"
    model_path = "static/model/bert_model.pb"
    max_seq_length = 512
    model.load_model(vocab_file, model_path, max_seq_length)
    model.predict_with_title("", "")


if __name__ == "__main__":
    init_model()
    app.run(port=8868)
