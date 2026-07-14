"""
中文语音识别系统 - Vosk + 自训练置信度校准网络
使用THCHS-30数据集训练
"""

import os
import json
import wave
import numpy as np
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from vosk import Model as VoskModel, KaldiRecognizer
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
VOSK_MODEL_PATH = r"D:\pycharm\人工智能\VoskModels\vosk-model-small-cn-0.22\vosk-model-small-cn-0.22"
THCHS30_PATH = r"D:\pycharm\人工智能\data_thchs30\data_thchs30"

class Config:
    sample_rate = 16000
    batch_size = 32
    epochs = 30
    lr = 0.001
    hidden_dim = 128
    max_train_samples = 500


# ==================== 数据加载（提取纯中文文本） ====================
def load_thchs30_data(data_path, max_samples=500):
    """加载THCHS-30数据集，提取纯中文文本"""
    data_dir = os.path.join(data_path, "data")

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在 - {data_dir}")
        return []

    data = []
    trn_files = [f for f in os.listdir(data_dir) if f.endswith('.wav.trn')]
    trn_files = trn_files[:max_samples]

    for trn_file in trn_files:
        wav_file = trn_file.replace('.trn', '')
        wav_path = os.path.join(data_dir, wav_file)
        trn_path = os.path.join(data_dir, trn_file)

        if os.path.exists(wav_path) and os.path.exists(trn_path):
            with open(trn_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # 提取纯中文文本（第一行，去掉拼音）
            chinese_text = content.split('\n')[0].strip().replace(' ', '')
            data.append((wav_path, chinese_text))

    print(f"成功加载 {len(data)} 条训练数据")
    if data:
        print(f"示例文本: {data[0][1][:50]}...")
    return data


# ==================== Vosk识别器 ====================
class VoskBaseRecognizer:
    def __init__(self, model_path=VOSK_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        self.model = VoskModel(model_path)
        self.sample_rate = Config.sample_rate

    def recognize(self, wav_path):
        wf = wave.open(wav_path, 'rb')
        rec = KaldiRecognizer(self.model, self.sample_rate)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        result = json.loads(rec.FinalResult())
        wf.close()

        return {
            'text': result.get('text', '').replace(' ', ''),  # 去掉空格
            'confidence': result.get('confidence', 0.0)
        }


# ==================== 置信度校准网络 ====================
class ConfidenceCalibrator(tf.keras.Model):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim // 2, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid')
        self.dropout = layers.Dropout(0.3)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.out(x)


# ==================== 构建训练数据 ====================
def build_training_data(data_list, vosk_recognizer):
    X = []
    y = []
    char_counter = Counter()

    for i, (wav_path, true_text) in enumerate(data_list):
        if i % 50 == 0:
            print(f"  处理进度: {i}/{len(data_list)}")

        vosk_res = vosk_recognizer.recognize(wav_path)
        hyp_text = vosk_res['text']
        confidence = vosk_res['confidence']
        true_text = true_text.replace(' ', '')

        # 标签: 1表示需要纠错, 0表示正确
        label = 0 if hyp_text == true_text else 1

        features = [
            confidence,
            len(true_text),
            len(hyp_text),
            1.0 if hyp_text and true_text and hyp_text[0] == true_text[0] else 0.0,
            abs(len(true_text) - len(hyp_text)) / max(len(true_text), 1)
        ]

        X.append(features)
        y.append(label)

        for c in true_text:
            char_counter[c] += 1

    print(f"特征提取完成，共 {len(X)} 条")
    print(f"需要纠错的样本比例: {sum(y)/len(y):.2%}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), char_counter


# ==================== 训练器 ====================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.confidence_net = None
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        self.history = None

    def build_vocab(self, char_counter, max_vocab=500):
        common_chars = [c for c, _ in char_counter.most_common(max_vocab - 2)]
        for i, c in enumerate(common_chars, start=len(self.char_to_idx)):
            self.char_to_idx[c] = i
            self.idx_to_char[i] = c
        print(f"词表大小: {len(self.char_to_idx)}")

    def build_models(self):
        self.confidence_net = ConfidenceCalibrator(self.config.hidden_dim)
        dummy_input = tf.zeros((1, 5))
        _ = self.confidence_net(dummy_input)

    def train_confidence_net(self, X, y, X_val, y_val):
        self.confidence_net.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = self.confidence_net.fit(
            X, y,
            validation_data=(X_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
        self.history = history.history
        return history

    def plot_history(self):
        """绘制训练曲线（只显示，不保存）"""
        if self.history is None:
            print("没有训练历史数据")
            return

        plt.figure(figsize=(14, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], 'b-', label='训练损失', linewidth=2)
        plt.plot(self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练与验证损失曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], 'b-', label='训练准确率', linewidth=2)
        plt.plot(self.history['val_accuracy'], 'r-', label='验证准确率', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('训练与验证准确率曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_models(self, path='./saved_models'):
        os.makedirs(path, exist_ok=True)
        self.confidence_net.save_weights(os.path.join(path, 'confidence_calibrator.weights.h5'))

        import pickle
        with open(os.path.join(path, 'vocab.pkl'), 'wb') as f:
            pickle.dump({'char_to_idx': self.char_to_idx, 'idx_to_char': self.idx_to_char}, f)

        print(f"模型已保存到 {path}")

    def load_models(self, path='./saved_models'):
        import pickle
        with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        self.char_to_idx = vocab['char_to_idx']
        self.idx_to_char = vocab['idx_to_char']

        self.confidence_net = ConfidenceCalibrator(self.config.hidden_dim)
        dummy_input = tf.zeros((1, 5))
        _ = self.confidence_net(dummy_input)
        self.confidence_net.load_weights(os.path.join(path, 'confidence_calibrator.weights.h5'))
        print(f"模型已从 {path} 加载")

    def predict(self, features):
        """预测是否需要纠错"""
        if self.confidence_net is None:
            raise ValueError("模型未加载")
        features = np.array(features).reshape(1, -1)
        prob = self.confidence_net.predict(features, verbose=0)[0][0]
        return prob, prob > 0.5


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("中文语音识别系统 - Vosk + 置信度校准网络")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载THCHS-30数据集")
    train_data = load_thchs30_data(THCHS30_PATH, Config.max_train_samples)
    if not train_data:
        print("请检查THCHS-30数据集路径")
        return

    # 划分训练/验证集
    split = int(0.8 * len(train_data))
    train_subset = train_data[:split]
    val_subset = train_data[split:]
    print(f"训练集: {len(train_subset)}条, 验证集: {len(val_subset)}条")

    # 2. 初始化Vosk
    print("\n[2/4] 初始化Vosk")
    vosk = VoskBaseRecognizer(VOSK_MODEL_PATH)

    # 3. 生成训练数据
    print("\n[3/4] 用Vosk生成训练数据...")
    X_train, y_train, char_counter = build_training_data(train_subset, vosk)
    X_val, y_val, _ = build_training_data(val_subset, vosk)
    print(f"特征维度: {X_train.shape[1]}")

    # 4. 训练
    print("\n[4/4] 训练置信度校准网络")
    trainer = Trainer(Config())
    trainer.build_vocab(char_counter)
    trainer.build_models()
    history = trainer.train_confidence_net(X_train, y_train, X_val, y_val)

    # 保存模型
    trainer.save_models()

    # 显示训练曲线
    print("\n[5/5] 显示训练曲线...")
    trainer.plot_history()

    print("\n✅ 完成！")
    print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()