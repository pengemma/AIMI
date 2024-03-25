import os
import pandas as pd

# 設定資料集根目錄
data_dir = 'chest_xray/chest_xray'

# 定義函數,用於生成CSV檔案
def generate_csv(folder_name):
    folder_path = os.path.join(data_dir, folder_name)
    entries = []
    
    # 遍歷資料夾內的每個類別資料夾
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            # 遍歷類別資料夾內的每個圖片檔案
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                LABEL = 1 if label == "PNEUMONIA" else 0
                entries.append((file_path, LABEL))
    
    # 建立DataFrame並存成CSV檔案
    df = pd.DataFrame(entries, columns=['path', 'label'])
    csv_path = os.path.join(data_dir, f'{folder_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f'CSV file saved: {csv_path}')

# 呼叫函數,生成train.csv、val.csv和test.csv
generate_csv('train')
generate_csv('val')
generate_csv('test')