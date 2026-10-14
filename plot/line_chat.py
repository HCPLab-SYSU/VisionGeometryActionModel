import matplotlib.pyplot as plt

def draw_custom_line_chart(
    data_list, 
    colors, 
    legends,
    font_size=12, 
    line_width=2, 
    fig_size=(8, 6), 
    marker_style='o',
    markersize=10,
    x_label="X Axis", 
    y_label="Y Axis"
):
    """
    封装好的折线图绘制函数
    
    :param data_list: 二维列表，包含n条长度为5的数据
    :param colors: 列表，包含n条线对应的颜色
    :param font_size: 全局字体大小
    :param line_width: 折线粗细
    :param fig_size: 画布尺寸 (宽, 高)
    :param marker_style: 折点样式 (如 'o', 's', 'D', 'x' 等)
    :param x_label: x轴名称
    :param y_label: y轴名称
    """
    
    # 1. 设置画布大小
    plt.figure(figsize=fig_size)
    
    # 2. 设置全局字体大小
    # 这里的设置会影响轴标签和刻度数字
    plt.rcParams.update({'font.size': font_size})
    
    # 3. 准备横坐标 (默认 1-5)
    x_values = [1, 2, 3, 4, 5]
    
    # 4. 循环绘制每一条线
    # zip 函数将数据和颜色一一对应
    for data, color, legend_text in zip(data_list, colors, legends):
        plt.plot(
            x_values, 
            data, 
            color=color, 
            linewidth=line_width, 
            marker=marker_style,
            markersize=markersize,
            label=legend_text
        )

    plt.legend(loc="lower right", prop={'size': 16})
    
    # 5. 设置坐标轴标签
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # 6. 设置刻度 (确保1-5的刻度都显示)
    plt.xticks(x_values)
    
    # 7. 移除标题和图例 (根据要求不调用 plt.title() 和 plt.legend())
    
    # 调整布局防止标签被遮挡
    plt.tight_layout()
    
    # 显示图像
    plt.savefig("aaa.png")



def main_draw():
    my_data = [
        [19.2, 80.4, 85.4, 89.4, 95.6],
        [10.0, 51.0, 76.0, 85.0, 88.4],
    ]
    
    # 准备对应的颜色
    my_colors = ['#c32b23', '#3d67a0']  # fc945d

    legends = ["w/ joint training", "w/o joint training"]
    
    # 调用函数
    draw_custom_line_chart(
        data_list=my_data,
        colors=my_colors,
        legends=legends,
        font_size=20,        # 字体大小
        line_width=5,        # 线宽
        fig_size=(5, 4),    # 画布宽10, 高5
        marker_style='o',    # 使用正方形折点
        markersize=10,
        x_label="Step (K)",
        y_label="Success Rate (%)"
    )

def ablation():
    my_data = [
        [19.2, 80.4, 85.4, 89.4, 95.6],
    ]
    
    # 准备对应的颜色
    my_colors = ['#fc945d',]

    legends = ["w/ joint training"]
    
    # 调用函数
    draw_custom_line_chart(
        data_list=my_data,
        colors=my_colors,
        legends=legends,
        font_size=20,        # 字体大小
        line_width=5,        # 线宽
        fig_size=(5, 4),    # 画布宽10, 高5
        marker_style='o',    # 使用正方形折点
        markersize=10,
        x_label="LoRA rank",
        y_label="Success Rate (%)"
    )


# --- 使用示例 ---
if __name__ == "__main__":
    main_draw()