# -*- coding: utf-8 -*-
"""
三维荧光光谱数据增强 - 理想条件版本
简化但高效的增强方案
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift
import random
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SimpleEEMAugmenter:
    """
    简化的三维荧光光谱数据增强器
    针对理想条件下的测量数据
    """

    def __init__(self):
        """初始化增强器"""
        self.noise_levels = [0.005, 0.01, 0.015]  # 噪声水平
        self.shift_ranges = [-2, -1, 0, 1, 2]  # 偏移范围

    def load_data(self, eem_matrices, concentrations):
        """
        直接加载数据
        Args:
            eem_matrices: 三维荧光光谱矩阵列表，形状为 (n_samples, n_ex, n_em)
            concentrations: 对应的浓度列表
        """
        self.original_eems = np.array(eem_matrices)
        self.original_concs = np.array(concentrations)
        self.n_samples, self.n_ex, self.n_em = self.original_eems.shape

        print(f"数据加载成功！")
        print(f"样本数量: {self.n_samples}")
        print(f"每个样本形状: ({self.n_ex}, {self.n_em})")
        print(f"浓度范围: {np.min(self.original_concs):.3f} - {np.max(self.original_concs):.3f}")

    def add_gaussian_noise(self, eem_matrix, noise_level=0.01):
        """
        添加高斯噪声
        """
        noise = np.random.randn(*eem_matrix.shape) * noise_level * np.max(eem_matrix)
        augmented = eem_matrix + noise
        augmented[augmented < 0] = 0
        return augmented

    def random_intensity_scale(self, eem_matrix, scale_range=(0.8, 1.2)):
        """
        随机强度缩放
        """
        scale = np.random.uniform(*scale_range)
        return eem_matrix * scale

    def wavelength_shift(self, eem_matrix, max_shift=2):
        """
        波长偏移
        """
        ex_shift = np.random.randint(-max_shift, max_shift + 1)
        em_shift = np.random.randint(-max_shift, max_shift + 1)

        shifted = np.roll(eem_matrix, shift=ex_shift, axis=0)
        shifted = np.roll(shifted, shift=em_shift, axis=1)

        # 处理边界
        if ex_shift > 0:
            shifted[:ex_shift, :] = 0
        elif ex_shift < 0:
            shifted[ex_shift:, :] = 0

        if em_shift > 0:
            shifted[:, :em_shift] = 0
        elif em_shift < 0:
            shifted[:, em_shift:] = 0

        return shifted

    def linear_mix(self, eem1, conc1, eem2, conc2):
        """
        线性混合两个样本
        """
        mix_ratio = np.random.uniform(0.2, 0.8)
        mixed_eem = mix_ratio * eem1 + (1 - mix_ratio) * eem2
        mixed_conc = mix_ratio * conc1 + (1 - mix_ratio) * conc2
        return mixed_eem, mixed_conc

    def random_gaussian_blur(self, eem_matrix, sigma_range=(0.5, 1.5)):
        """
        随机高斯模糊（模拟分辨率变化）
        """
        sigma = np.random.uniform(*sigma_range)
        blurred = gaussian_filter(eem_matrix, sigma=sigma)
        return blurred

    def generate_augmented_batch(self, eem_matrix, concentration, n_augment=5):
        """
        为单个样本生成增强批次
        """
        augmented_eems = []
        augmented_concs = []

        for _ in range(n_augment):
            aug_eem = eem_matrix.copy()

            # 随机应用增强方法
            methods = []

            # 噪声增强
            if np.random.rand() > 0.3:
                noise_level = random.choice(self.noise_levels)
                aug_eem = self.add_gaussian_noise(aug_eem, noise_level)
                methods.append('noise')

            # 强度缩放
            if np.random.rand() > 0.3:
                aug_eem = self.random_intensity_scale(aug_eem)
                methods.append('intensity')

            # 波长偏移
            if np.random.rand() > 0.5:
                aug_eem = self.wavelength_shift(aug_eem, max_shift=2)
                methods.append('shift')

            # 高斯模糊
            if np.random.rand() > 0.7:
                aug_eem = self.random_gaussian_blur(aug_eem)
                methods.append('blur')

            augmented_eems.append(aug_eem)
            augmented_concs.append(concentration)

        return augmented_eems, augmented_concs

    def augment_dataset(self, augmentation_factor=20):
        """
        增强整个数据集
        Args:
            augmentation_factor: 总增强倍数
        Returns:
            增强后的数据集
        """
        print(f"\n开始数据增强...")
        print(f"目标增强倍数: {augmentation_factor}倍")

        augmented_eems = []
        augmented_concs = []

        # 添加原始数据
        augmented_eems.extend(self.original_eems)
        augmented_concs.extend(self.original_concs)

        # 计算每个原始样本需要生成的增强样本数
        samples_per_original = augmentation_factor - 1

        for i, (eem, conc) in enumerate(zip(self.original_eems, self.original_concs)):
            if i % 10 == 0:
                print(f"正在增强第 {i + 1}/{self.n_samples} 个样本...")

            # 为每个样本生成增强数据
            aug_eems, aug_concs = self.generate_augmented_batch(eem, conc, samples_per_original)
            augmented_eems.extend(aug_eems)
            augmented_concs.extend(aug_concs)

        # 转换为numpy数组
        augmented_eems = np.array(augmented_eems)
        augmented_concs = np.array(augmented_concs)

        print(f"\n增强完成！")
        print(f"原始数据量: {self.n_samples}")
        print(f"增强后数据量: {len(augmented_eems)}")
        print(f"实际增强倍数: {len(augmented_eems) / self.n_samples:.1f}倍")

        return augmented_eems, augmented_concs

    def visualize_augmentation(self, n_examples=3):
        """
        可视化增强效果
        """
        fig, axes = plt.subplots(n_examples, 5, figsize=(15, 3 * n_examples))

        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            if i >= self.n_samples:
                break

            original = self.original_eems[i]

            # 显示原始数据
            im1 = axes[i, 0].imshow(original, aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'原始样本 {i + 1}')
            axes[i, 0].set_xlabel('发射')
            axes[i, 0].set_ylabel('激发')

            # 生成并显示4个增强样本
            aug_eems, _ = self.generate_augmented_batch(original, 0, 4)

            for j in range(4):
                im = axes[i, j + 1].imshow(aug_eems[j], aspect='auto', cmap='viridis')
                axes[i, j + 1].set_title(f'增强 {j + 1}')
                axes[i, j + 1].set_xlabel('发射')
                axes[i, j + 1].set_ylabel('激发')

        plt.tight_layout()
        plt.savefig('augmentation_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"增强可视化已保存到: augmentation_visualization.png")


# 二、数据预处理和模型训练
def train_with_augmented_data(augmented_eems, augmented_concs, test_size=0.2, epochs=50):
    """
    使用增强数据训练模型
    """
    print("\n" + "=" * 50)
    print("开始模型训练...")

    # 1. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        augmented_eems, augmented_concs,
        test_size=test_size, random_state=42
    )

    # 进一步分割验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # 2. 数据归一化
    def normalize_eem(eem_data):
        """对每个样本进行最大归一化"""
        normalized = np.zeros_like(eem_data)
        for i in range(len(eem_data)):
            max_val = np.max(eem_data[i])
            if max_val > 0:
                normalized[i] = eem_data[i] / max_val
        return normalized

    X_train_norm = normalize_eem(X_train)
    X_val_norm = normalize_eem(X_val)
    X_test_norm = normalize_eem(X_test)

    # 3. 浓度归一化
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    # 4. 创建PyTorch数据集
    class EEMDataset(Dataset):
        def __init__(self, eem_data, concentrations):
            self.eem_data = torch.FloatTensor(np.expand_dims(eem_data, axis=1))
            self.concentrations = torch.FloatTensor(concentrations)

        def __len__(self):
            return len(self.eem_data)

        def __getitem__(self, idx):
            return self.eem_data[idx], self.concentrations[idx]

    train_dataset = EEMDataset(X_train_norm, y_train_norm)
    val_dataset = EEMDataset(X_val_norm, y_val_norm)
    test_dataset = EEMDataset(X_test_norm, y_test_norm)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. 定义简单CNN模型
    class SimpleEEMModel(torch.nn.Module):
        def __init__(self, input_height, input_width):
            super(SimpleEEMModel, self).__init__()

            self.conv_layers = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),

                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )

            # 计算卷积后的尺寸
            with torch.no_grad():
                dummy = torch.zeros(1, 1, input_height, input_width)
                conv_out = self.conv_layers(dummy)
                self.conv_output_size = conv_out.numel()

            self.fc_layers = torch.nn.Sequential(
                torch.nn.Linear(self.conv_output_size, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),

                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),

                torch.nn.Linear(32, 1),
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

    # 6. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    n_ex, n_em = X_train_norm.shape[1], X_train_norm.shape[2]
    model = SimpleEEMModel(n_ex, n_em).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 7. 评估模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 在测试集上评估
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_predictions.extend(outputs.cpu().numpy().flatten())
            test_targets.extend(batch_y.numpy().flatten())

    # 反归一化
    test_predictions = np.array(test_predictions) * y_std + y_mean
    test_targets = np.array(test_targets) * y_std + y_mean

    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(test_targets, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)

    print("\n" + "=" * 50)
    print("模型性能评估:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print("=" * 50)

    # 8. 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 预测 vs 真实
    axes[0].scatter(test_targets, test_predictions, alpha=0.6)
    axes[0].plot([test_targets.min(), test_targets.max()],
                 [test_targets.min(), test_targets.max()], 'r--', lw=2)
    axes[0].set_xlabel('真实浓度')
    axes[0].set_ylabel('预测浓度')
    axes[0].set_title('预测 vs 真实')
    axes[0].grid(True, alpha=0.3)

    # 残差图
    residuals = test_predictions - test_targets
    axes[1].scatter(test_targets, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('真实浓度')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差图')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, test_predictions, test_targets


# 三、主程序
def main():
    """
    主程序
    """
    print("=" * 60)
    print("三维荧光光谱数据增强与抗生素浓度预测")
    print("=" * 60)

    # 1. 准备数据（这里用模拟数据演示，请替换为您的数据）
    print("\n1. 准备数据...")

    # 创建模拟数据
    n_samples = 120
    n_ex, n_em = 50, 50

    # 模拟三维荧光光谱
    original_eems = []
    original_concs = []

    for i in range(n_samples):
        # 创建基础光谱
        eem = np.zeros((n_ex, n_em))

        # 添加几个荧光峰
        for _ in range(np.random.randint(2, 4)):
            center_ex = np.random.randint(10, 40)
            center_em = np.random.randint(10, 40)
            intensity = np.random.uniform(0.5, 1.0)
            width_ex = np.random.uniform(3, 6)
            width_em = np.random.uniform(3, 6)

            x = np.arange(n_ex)
            y = np.arange(n_em)
            X, Y = np.meshgrid(x, y, indexing='ij')

            peak = intensity * np.exp(-((X - center_ex) ** 2 / (2 * width_ex ** 2) +
                                        (Y - center_em) ** 2 / (2 * width_em ** 2)))
            eem += peak

        # 添加背景
        eem += np.random.uniform(0.05, 0.1)

        original_eems.append(eem)

        # 浓度与总荧光强度相关
        total_intensity = np.sum(eem)
        concentration = total_intensity * np.random.uniform(0.8, 1.2)
        original_concs.append(concentration)

    # 2. 数据增强
    print("\n2. 数据增强...")

    augmenter = SimpleEEMAugmenter()
    augmenter.load_data(original_eems, original_concs)

    # 可视化增强效果
    augmenter.visualize_augmentation(n_examples=3)

    # 执行增强（20倍增强：120 → 约2400个样本）
    augmented_eems, augmented_concs = augmenter.augment_dataset(augmentation_factor=20)

    # 保存增强后的数据
    np.savez_compressed('augmented_data.npz',
                        eems=augmented_eems,
                        concentrations=augmented_concs)
    print(f"\n增强数据已保存到: augmented_data.npz")

    # 3. 模型训练
    print("\n3. 模型训练...")

    model, predictions, targets = train_with_augmented_data(
        augmented_eems,
        augmented_concs,
        test_size=0.2,
        epochs=50
    )

    # 4. 保存完整结果
    print("\n4. 保存结果...")

    results = {
        'n_original_samples': n_samples,
        'n_augmented_samples': len(augmented_eems),
        'augmentation_factor': len(augmented_eems) / n_samples,
        'model_path': 'best_model.pth',
        'data_path': 'augmented_data.npz',
    }

    import json
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("训练结果已保存到: training_results.json")
    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)

    return augmenter, model, augmented_eems, augmented_concs


# 四、快速使用函数
def quick_augment_and_train(eem_matrices, concentrations, augmentation_factor=20):
    """
    快速增强和训练函数
    只需传入您的数据和浓度即可
    """
    # 1. 初始化增强器
    augmenter = SimpleEEMAugmenter()
    augmenter.load_data(eem_matrices, concentrations)

    # 2. 执行增强
    print(f"\n执行 {augmentation_factor} 倍数据增强...")
    augmented_eems, augmented_concs = augmenter.augment_dataset(augmentation_factor)

    # 3. 训练模型
    print(f"\n开始模型训练...")
    model, predictions, targets = train_with_augmented_data(
        augmented_eems,
        augmented_concs
    )

    return augmenter, model, augmented_eems, augmented_concs


# 五、使用示例
if __name__ == "__main__":
    """
    使用说明：

    如果您有自己的数据：
    1. 准备 eem_matrices 和 concentrations
    2. 调用 quick_augment_and_train(eem_matrices, concentrations)

    示例：
    augmenter, model, augmented_data, augmented_labels = quick_augment_and_train(
        your_eem_data,  # 形状: (n_samples, n_ex, n_em)
        your_concentrations,  # 形状: (n_samples,)
        augmentation_factor=20  # 增强倍数
    )
    """

    # 演示运行
    main()