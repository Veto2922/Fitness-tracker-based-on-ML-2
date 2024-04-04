import requests

acc_path = './data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv'
gyr_path = './data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv'



print(open(acc_path, 'rb'))

files={
    'acc': open(acc_path, 'rb'),
    'gyr': open(gyr_path, 'rb')
}
# Define the form data
req = requests.post('http://127.0.0.1:5000/predict',files=files)
print(req.text)