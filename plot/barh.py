"""
作横着长的柱状图
"""

import matplotlib.pyplot as plt

def plot_horizontal_bars(data, colors, bar_height=0.5, spacing=0.2, tick_fontsize=10, figsize=(10, 6)):
    """
    绘制自定义横向柱状图
    
    :param data: 数值列表 (n个数据)
    :param colors: 颜色列表 (n个颜色)
    :param bar_height: 柱子的粗细 (对应通常理解的宽度)
    :param spacing: 柱子之间的间距
    :param tick_fontsize: 刻度文字的大小
    :param figsize: 画布尺寸 (宽, 高)
    """
    n = len(data)
    
    # 计算 y 轴坐标：每个柱子的中心位置
    # 步长 = 柱子粗细 + 间距
    step = bar_height + spacing
    y_positions = [i * step for i in range(n)]
    
    # 1. 创建 figure 和 axes，设置尺寸
    fig, ax = plt.subplots(figsize=figsize)
    
    # 2. 绘制横向柱状图
    # 在 barh 中，height 参数控制柱子的“粗细”
    ax.barh(y_positions, data, color=colors, height=bar_height)
    
    # 3. 设置刻度文字字号
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # 4. 去除刻度线 (Ticks)，保留刻度文字 (Labels)
    ax.tick_params(axis='both', which='both', length=0)
    
    # 5. 移除不需要的元素（图例、坐标轴名、标题已经在默认中不显示，这里确保简洁）
    # 如果需要把四周的边框也去掉，可以取消下面代码的注释：
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    plt.savefig("aaa.png")

# --- 使用示例 ---
if __name__ == "__main__":
    my_data = [98.0, 95.0]
    my_colors = ['#FF6B6B', '#4ECDC4']
    
    plot_horizontal_bars(
        data=my_data, 
        colors=my_colors, 
        bar_height=0.6,    # 柱子粗细
        spacing=0.4,       # 柱子间距
        tick_fontsize=12,  # 刻度字号
        figsize=(3, 3)     # 图像尺寸
    )