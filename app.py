import csv
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
app = Flask(__name__)

def loadData(path):
    f = open(path, "r")
    data = csv.reader(f)
    data = np.array(list(data))
    data = np.delete(data, 0, 0)
    np.random.shuffle(data)
    f.close()
    return data

dulieuLoad = loadData("x_resampled.csv")
dulieu_X = dulieuLoad[:, :-1].astype(np.float64)  # Lấy tất cả các cột ngoại trừ cột cuối cùng
dulieu_Y = dulieuLoad[:, -1]   # Lấy cột cuối cùng làm dữ liệu nhãn

#Để lấy các danh sách thuộc tính
data1=pd.read_csv("bank.csv",delimiter=';')
n=len(data1.columns)
ds_thuoctinh = list(data1.iloc[:,0:n-1].columns)

model_bayes = joblib.load('model_bayes.pkl')
model_dt = joblib.load('model_dt.pkl')
model_rf = joblib.load('model_rf.pkl')
model_bagging = joblib.load('model_bagging.pkl')

selected_model = model_bayes  # Khởi tạo selected_model với giá trị mặc định

@app.route('/')
def hello_world():
    return render_template('index.html', attributes=ds_thuoctinh)

@app.route('/predict', methods=['POST'])
def predict():
    global selected_model
    # Lấy dữ liệu từ form
    algorithm = request.form['algorithm']

    # # Tạo mô hình dựa trên lựa chọn của người dùng
    if algorithm == 'gaussian_naive_bayes':
        selected_model = model_bayes
    elif algorithm == 'decision_tree':
        selected_model = model_dt
    elif algorithm == 'random_forest':
        selected_model = model_rf
    elif algorithm == 'bagging':
        selected_model = model_bagging
    
    attributes = []
    for i in range(16):
        # attribute = float(request.form[f'attribute_{i+1}'])
        attribute = request.form[f'attribute_{i+1}']
        attributes.append(attribute)

    attributes = ['' if attr == '' else int(attr) for attr in attributes]
    # attributes = [float(attr) for attr in attributes]
    print("attributes", attributes)
    # Chuyển dữ liệu đầu vào thành một DataFrame Pandas với tên cột tương ứng
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    df = pd.DataFrame([attributes], columns=columns)

    data_min = np.array([18, 0, 0, 0, 0, -8019, 0, 0, 0, 1, 0, 0, 1, -1, 0, 0])
    data_max = np.array([95, 11, 2, 3, 1, 102127, 1, 1, 2, 31, 11, 4918, 63, 871, 275, 3])

    # Khởi tạo scaler và fit với dữ liệu huấn luyện
    scaler = MinMaxScaler()
    scaler.fit(dulieu_X)  # dulieu_X là dữ liệu đã được chuẩn hóa

    # Chuẩn hóa dữ liệu mới
    # scaled_data = scaler.transform(df)
    scaled_data = (df - data_min) / (data_max - data_min)

    print('scaled_data', scaled_data)

    # Dự đoán giá trị 'y' (có hoặc không đăng ký gói tiết kiệm)
    prediction = selected_model.predict(scaled_data)

    # prediction = selected_model.predict([scaled_data])
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
