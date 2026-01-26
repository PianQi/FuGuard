#%% ------------------ Imports & CONFIG ------------------
import numpy as np
import json
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from FuGuard.main_AWE.config import CONFIG


os.makedirs(CONFIG["SAVE_FOLDER"], exist_ok=True)
rng = np.random.default_rng(CONFIG["SEED"])

# ------------------ 状态/训练/测试组（保持硬编码） ------------------
STATE_GROUPS = {0:7,1:8,2:8,3:8,4:8,5:8}
TRAIN_GROUPS = {
    0:[0,1,2,3,4,5],
    1:[0,1,2,3,4,5,6],
    2:[0,1,2,3,4,5,6],
    3:[0,1,2,3,4,5,6],
    4:[0,1,2,3,4,5,6],
    5:[0,1,2,3,4,5,6]
}
TEST_GROUPS = {0:[6],1:[7],2:[7],3:[7],4:[7],5:[7]}
PRIMARY_GROUPS = {0:[0,1,2,3],1:[0,1,2,3],2:[0,1,2,3],3:[0,1,2,3],4:[0,1,2,3],5:[0,1,2,3]}
STATE_TO_CLIENT = {0:"client_0", 1:"client_1", 2:"client_2", 3:"client_3", 4:"client_4", 5:"client_5"}

#%% ------------------ 数据处理函数 ------------------
def load_dict(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_data(filename):
    data_dict = load_dict(filename)
    vibr_UB, vibr_LB = [], []
    for j in range(len(data_dict)):
        vibr_UB.append(data_dict[str(j)]['vibr_UB'])
        vibr_LB.append(data_dict[str(j)]['vibr_LB'])
    return np.c_[vibr_UB, vibr_LB]

def zscore_normalize(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / std

def lowpass_iir(data, cutoff=CONFIG["LOWPASS_CUTOFF"], fs=CONFIG["FS_ORIG"], order=CONFIG["FILTER_ORDER"]):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=1)

def downsample(data, factor=CONFIG["FS_ORIG"]//CONFIG["FS_TARGET"]):
    return data[:, ::factor]

def create_windows(data, window_size=CONFIG["WINDOW_SIZE"], stride=CONFIG["STRIDE"]):
    windows = []
    for start in range(0, data.shape[0] - window_size + 1, stride):
        windows.append(data[start:start+window_size, :])
    return np.array(windows)

def generate_labels(num_windows, state):
    return np.full(num_windows, state)


# backdoor
def add_spike_trigger(
    X_np,
    channel_idx=CONFIG["BACKDOOR_CHANNEL_IDX"],
    time_idx=CONFIG["BACKDOOR_TIME_IDX"],
    amplitude=CONFIG["BACKDOOR_AMPLITUDE"],
    extra_channels=None,
    extra_times=None,
    noise_std=0.0
):
    """
    增强版 spike trigger 注入（所有样本都注入）
    
    Args:
        X_np: numpy array, shape [N, C, T]
        channel_idx: 主触发通道
        time_idx: 主触发时间点
        amplitude: spike 幅值
        extra_channels: list of int, 额外触发通道
        extra_times: list of int, 额外触发时间点
        noise_std: float, 高斯噪声标准差
    
    Returns:
        X_poisoned: numpy array, same shape as X_np
    """
    X = X_np.copy()
    N, C, T = X.shape

    # 所有样本都注入主触发器
    X[:, channel_idx, time_idx] += amplitude

    # 额外通道触发
    if extra_channels is not None:
        for ch in extra_channels:
            X[:, ch, time_idx] += amplitude

    # 额外时间点触发
    if extra_times is not None:
        for t in extra_times:
            X[:, channel_idx, t] += amplitude
            if extra_channels is not None:
                for ch in extra_channels:
                    X[:, ch, t] += amplitude

    # 高斯噪声
    if noise_std > 0.0:
        X += np.random.normal(0, noise_std, X.shape)

    return X


#%% ------------------ 客户端数据初始化 ------------------
client_data = {c: {"train": [], "train_labels": []} for c in CONFIG["CLIENTS"]}
global_test_set, global_test_labels = [], []

client_data_clean = {c: {"train": [], "train_labels": []} for c in CONFIG["CLIENTS"]}

for state in STATE_GROUPS.keys():
    for group in TRAIN_GROUPS[state]:
        filename = os.path.join(CONFIG["DATA_FOLDER"], f"{state}chips", f"{state}chips_{group}.json")
        if not os.path.exists(filename):
            continue

        data = load_data(filename)
        data = zscore_normalize(data)
        data = lowpass_iir(data)
        data = downsample(data)

        train_windows = create_windows(data)
        train_labels = generate_labels(len(train_windows), state)

        if group in PRIMARY_GROUPS[state]:
            client = STATE_TO_CLIENT[state]
        else:
            client = rng.choice(CONFIG["CLIENTS"])

        client_data_clean[client]["train"].append(train_windows)
        client_data_clean[client]["train_labels"].append(train_labels)

        # 后门注入
        if client == CONFIG["BACKDOOR_CLIENT"]:

            # 可选增强版调用：多通道、多时间点、高斯噪声
            train_windows = add_spike_trigger(
                train_windows,
                amplitude=20,                     # 增大幅值
                extra_channels=[1,2],             # 额外通道
                extra_times=[5,6,7],                # 额外时间点
                noise_std=0.5                     # 高斯噪声增强
            )

            train_labels = np.full_like(train_labels, CONFIG["BACKDOOR_TARGET_CLASS"])

        client_data[client]["train"].append(train_windows)
        client_data[client]["train_labels"].append(train_labels)

    for group in TEST_GROUPS[state]:
        filename = os.path.join(CONFIG["DATA_FOLDER"], f"{state}chips", f"{state}chips_{group}.json")
        if not os.path.exists(filename):
            continue

        data = load_data(filename)
        data = zscore_normalize(data)
        data = lowpass_iir(data)
        data = downsample(data)
        windows = create_windows(data)
        labels = generate_labels(len(windows), state)

        global_test_set.append(windows)
        global_test_labels.append(labels)

#%% ------------------ 拼接客户端训练/验证集 ------------------
for client in CONFIG["CLIENTS"]:
    client_data[client]["train_X"] = np.concatenate(client_data[client]["train"], axis=0)
    client_data[client]["train_y"] = np.concatenate(client_data[client]["train_labels"], axis=0)
    print(f"{client} 总训练样本数={client_data[client]['train_X'].shape[0]}")

#%% ------------------ 拼接全局测试集 ------------------
global_test_X = np.concatenate(global_test_set, axis=0)
global_test_y = np.concatenate(global_test_labels, axis=0)

mask = (global_test_y != CONFIG["BACKDOOR_TARGET_CLASS"])
# X_triggered = add_spike_trigger(global_test_X[mask])
X_triggered = add_spike_trigger(
    global_test_X[mask],
                amplitude=20,                     # 增大幅值
                extra_channels=[1,2],             # 额外通道
                extra_times=[5,6,7],                # 额外时间点
                noise_std=0.5                     # 高斯噪声增强
)
y_triggered = np.full(len(X_triggered), CONFIG["BACKDOOR_TARGET_CLASS"], dtype=global_test_y.dtype)

#%% ------------------ 保存处理后的数据 ------------------
for client in CONFIG["CLIENTS"]:
    np.savez(os.path.join(CONFIG["SAVE_FOLDER"], f"{client}_train.npz"),
             train_X=client_data[client]["train_X"],
             train_y=client_data[client]["train_y"])

np.savez(os.path.join(CONFIG["SAVE_FOLDER"], "global_test.npz"),
         X_clean=global_test_X,
         y_clean=global_test_y,
         X_backdoor=X_triggered,
         y_backdoor=y_triggered)

print("客户端训练集和全局测试集（干净+后门）已保存完成！")
#%%
# ---------- 客户端训练/验证集数据分布 ----------
def plot_client_data_distribution(client_data, clients, subset="train"):
    """
    可视化每个客户端的数据分布。
    subset: "train"
    """
    key_y = f"{subset}_y"
    all_labels = sorted({label for client in clients for label in np.unique(client_data[client][key_y])})
    bar_width = 0.6
    colors = plt.cm.tab10.colors
    client_indices = np.arange(len(clients))
    bottoms = np.zeros(len(clients))

    plt.figure(figsize=(10,6))
    for i, label in enumerate(all_labels):
        counts = []
        for client in clients:
            y = client_data[client][key_y]
            counts.append(np.sum(y == label))
        counts = np.array(counts)
        plt.bar(client_indices, counts, bottom=bottoms, width=bar_width,
                color=colors[i % 10], label=f"class {label}")
        bottoms += counts

    plt.xticks(client_indices, clients)
    plt.ylabel(f"Number of {subset} samples")
    plt.title(f"Client {subset.capitalize()} Sample Distribution by Class")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_client_data_distribution(client_data, CONFIG["CLIENTS"], subset="train")


# ---------- 全局测试集分布 ----------
def plot_global_test_distribution(global_test_X, global_test_y, X_backdoor, y_backdoor):
    unique_clean, counts_clean = np.unique(global_test_y, return_counts=True)
    unique_bd, counts_bd = np.unique(y_backdoor, return_counts=True)

    plt.figure(figsize=(10,5))
    plt.bar(unique_clean - 0.2, counts_clean, width=0.4, label="Clean Test", color='lightgreen')
    plt.bar(unique_bd + 0.2, counts_bd, width=0.4, label="Backdoor Test", color='salmon')
    plt.xlabel("Class Label")
    plt.ylabel("Number of samples")
    plt.title("Global Test Set Distribution (Clean vs Backdoor)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_global_test_distribution(global_test_X, global_test_y, X_triggered, y_triggered)

# %%
def plot_backdoor_signal(client_name, window_idx=0, channel_idx=0, time_length=30,
                         client_data_clean=None, client_data=None, train_idx=0):
    """
    绘制指定客户端训练窗口的原始信号与后门信号对比。
    
    参数：
        client_name: str, 客户端名称
        window_idx: int, 窗口索引
        channel_idx: int, 通道索引
        time_length: int, 显示时间步长度
        client_data_clean: dict, 包含干净数据
        client_data: dict, 包含后门数据
        train_idx: int, train list 中的索引
    """
    if client_data_clean is None or client_data is None:
        raise ValueError("请提供 client_data_clean 和 client_data")

    # 干净信号
    clean_array = client_data_clean[client_name]["train"][train_idx]
    clean_window = clean_array[window_idx, :time_length, channel_idx]

    # 后门信号
    backdoor_array = client_data[client_name]["train"][train_idx]
    backdoor_window = backdoor_array[window_idx, :time_length, channel_idx]

    plt.figure(figsize=(10,4))
    plt.plot(clean_window, label="Original Signal", color='blue')
    plt.plot(backdoor_window, label="Backdoor Signal", color='red', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.title(f"{client_name} Window {window_idx} - Signal Before/After Backdoor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_backdoor_signal(
    client_name='client_2',
    window_idx=0,
    channel_idx=0,
    time_length=30,
    client_data_clean=client_data_clean,
    client_data=client_data,
    train_idx=0
)

#%%

def plot_backdoor_signal_heatmap(client_name, window_idx=0, time_length=None,
                                 client_data_clean=None, client_data=None,
                                 train_idx=0, num_channels=3, cmap_clean='plasma_r', cmap_backdoor='plasma_r'):
    """
    绘制指定客户端训练窗口的多通道信号热力图，显示原始信号与后门信号对比。
    """
    if client_data_clean is None or client_data is None:
        raise ValueError("请提供 client_data_clean 和 client_data")
    
    clean_array = client_data_clean[client_name]["train"][train_idx][window_idx]
    backdoor_array = client_data[client_name]["train"][train_idx][window_idx]
    
    if time_length is None:
        time_length = clean_array.shape[0]

    # 取前 num_channels 个通道和前 time_length 个时间步
    Z_clean = np.array([clean_array[:time_length, ch] for ch in range(num_channels)])
    Z_backdoor = np.array([backdoor_array[:time_length, ch] for ch in range(num_channels)])
    
    # 确定颜色范围一致，便于对比
    vmin = min(Z_clean.min(), Z_backdoor.min())
    vmax = max(Z_clean.max(), Z_backdoor.max())

    fig, axs = plt.subplots(1, 2, figsize=(7, 3),dpi=500)

    im1 = axs[0].imshow(Z_clean, aspect='auto', origin='lower', cmap=cmap_clean, vmin=vmin, vmax=vmax)
    axs[0].set_title("Clean Signal", fontsize=12)
    axs[0].set_xlabel("Time Step", fontsize=12)
    axs[0].set_ylabel("Feature Index", fontsize=12)
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(Z_backdoor, aspect='auto', origin='lower', cmap=cmap_backdoor, vmin=vmin, vmax=vmax)
    axs[1].set_title("Triggered Signal", fontsize=12)
    axs[1].set_xlabel("Time Step", fontsize=12)
    axs[1].set_ylabel("Feature Index", fontsize=12)
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()


plot_backdoor_signal_heatmap(
    client_name='client_2',
    window_idx=0,
    time_length=100,
    client_data_clean=client_data_clean,
    client_data=client_data,
    train_idx=0,
    num_channels=500
)

# %%
