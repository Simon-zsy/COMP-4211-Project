import matplotlib.pyplot as plt
from PIL import Image

# 读取归一化坐标
def read_normalized_points(file_path):
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        points = list(map(float, line.split()))
    return [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

# 在图像上绘制点
def draw_ground_truth(image_path, points, output_path):
    image = Image.open(image_path)
    plt.imshow(image)
    
    # 绘制点
    for (x, y) in points:
        plt.scatter(x, y, color='blue', s=50)  # s 为点的大小

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')  # 保存图像
    plt.close()  # 关闭图像

if __name__ == "__main__":
    points_file = '/localdata/szhoubx/rm/data/rune_blender_v0.2/crop_labels/rune_0.5_(0, -1, -1, -1, -1)_1_000248_class_1_conf_0.96_resized.txt'  # 替换为你的点文件路径
    image_path = '/localdata/szhoubx/rm/data/rune_blender_v0.2/crop/rune_0.5_(0, -1, -1, -1, -1)_1_000248_class_1_conf_0.96_resized.jpg'  # 替换为你的图像路径
    output_image_path = './'  # 替换为保存路径

    # 读取归一化坐标
    normalized_points = read_normalized_points(points_file)

    # 你可能需要将归一化坐标转换为图像坐标
    # 假设图像尺寸为 (width, height)
    image = Image.open(image_path)
    width, height = image.size
    image_points = [(x * width, y * height) for (x, y) in normalized_points]
    print(image.size)

    # 绘制并保存图像
    draw_ground_truth(image_path, image_points, output_image_path)