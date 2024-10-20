import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

# 判断是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载和预处理模块
class DataPreprocessor:
    def __init__(self, filepath, seq_length=96):
        self.filepath = filepath
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        # 加载数据
        df = pd.read_csv(self.filepath, parse_dates=['date'])

        # 数据标准化
        features = ['Temp', 'Humidity', 'GHI', 'SHI', 'Power']
        df[features] = self.scaler.fit_transform(df[features])

        # 创建输入序列和目标
        data = df[features].values
        x, y = [], []
        for i in range(len(data) - self.seq_length):
            x.append(data[i:i + self.seq_length, :-1])  # 输入：气象数据
            y.append(data[i + self.seq_length, -1])  # 目标：发电功率

        return np.array(x), np.array(y), df['date'].values[self.seq_length:]  # 返回日期


# 模型定义模块
class LSTMPVModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMPVModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取LSTM输出的最后一个时间步
        return out


# 训练和评估模块
class Trainer:
    def __init__(self, model, lr, epochs):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.MSELoss()  # 使用MSE损失
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_loader):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # 调整目标 y_batch 的形状
                y_batch = y_batch.view(-1, 1)

                # 前向传播
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # 打印每轮的训练损失
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {train_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # 调整目标 y_batch 的形状
                y_batch = y_batch.view(-1, 1)

                outputs = self.model(x_batch)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = math.sqrt(mse)

        return mae, mse, rmse, predictions, actuals


# 可视化和保存结果模块
class ResultsSaver:
    def __init__(self, filepath):
        self.filepath = filepath

    def save_results(self, dates, predictions, actuals):
        # 仅保存未来24小时的预测结果
        predictions = predictions.ravel()[:96]  # 取前96个预测值
        actuals = actuals.ravel()[:96]  # 取前96个实际值
        dates = dates[:96]  # 取前96个日期

        # 创建 DataFrame 并保存
        result_df = pd.DataFrame({
            'Date': dates,
            'Predicted Power': predictions,
            'Actual Power': actuals
        })
        result_df.to_csv(self.filepath, index=False)

    def plot_results(self, dates, predictions, actuals, plot_filepath):
        # 仅绘制未来24小时的预测结果
        predictions = predictions.ravel()[:96]  # 取前96个预测值
        actuals = actuals.ravel()[:96]  # 取前96个实际值
        dates = dates[:96]  # 取前96个日期

        plt.figure(figsize=(10, 6))
        plt.plot(range(96), actuals, label='Actual Power', color='blue')  # 横坐标为时间步
        plt.plot(range(96), predictions, label='Predicted Power', color='red')  # 横坐标为时间步
        plt.xlabel('Time Steps (15 min intervals)')
        plt.ylabel('Power')
        plt.title('Actual vs Predicted Power for Next 24 Hours')
        plt.legend()
        plt.savefig(plot_filepath)
        plt.show()


# 主程序模块
def main():
    # 文件路径
    data_filepath = 'Sanyo.csv'  # 你的数据集路径
    result_filepath = 'prediction_results1.csv'
    plot_filepath = 'prediction_plot1.png'

    # 参数设置
    seq_length = 96  # 24小时数据，15分钟间隔
    batch_size = 64
    hidden_size = 128
    num_layers = 4
    dropout = 0.2
    learning_rate = 0.001
    epochs = 100

    # 1. 数据加载和预处理
    preprocessor = DataPreprocessor(data_filepath, seq_length)
    x, y, dates = preprocessor.load_and_preprocess_data()

    # 2. 数据集划分
    total_samples = len(x)
    train_size = int(0.7 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size

    x_train, x_val, x_test = x[:train_size], x[train_size:train_size + val_size], x[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # 转换为张量
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. 模型定义
    input_size = x_train.shape[2]
    model = LSTMPVModel(input_size, hidden_size, num_layers, dropout)

    # 4. 模型训练
    trainer = Trainer(model, learning_rate, epochs)
    trainer.train(train_loader)

    # 5. 模型评估
    mae, mse, rmse, predictions, actuals = trainer.evaluate(test_loader)
    print(f"MAE: {mae:.8f}, MSE: {mse:.8f}, RMSE: {rmse:.8f}")

    # 6. 保存结果并可视化
    test_dates = dates[-len(actuals):]  # 取测试集对应的日期
    results_saver = ResultsSaver(result_filepath)
    results_saver.save_results(test_dates, predictions, actuals)
    results_saver.plot_results(test_dates, predictions, actuals, plot_filepath)


if __name__ == "__main__":
    main()
