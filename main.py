import numpy as np
import matplotlib.pyplot as plt


def read_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) == 0:
                continue
            if elems[0] == 'v':
                x, y = map(float, elems[1:3])
                vertices.append([x, y])
            elif elems[0] == 'f':
                indices = [int(index) - 1 for index in elems[1:]]
                faces.append(indices)

    return np.array(vertices), np.array(faces)


def scale_matrix(scale_x, scale_y):
    """Матрица масштабирования"""
    return np.array([[scale_x, 0, 0],
                     [0, scale_y, 0],
                     [0, 0, 1]])


def translation_matrix(translation_x, translation_y):
    """Матрица переноса"""
    return np.array([[1, 0, translation_x],
                     [0, 1, translation_y],
                     [0, 0, 1]])


def scale_and_translate(vertices, image_width, image_height):
    # нахождение диапазона координат
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    diff_x = max_x - min_x
    center_x = (max_x + min_x) / 2

    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    diff_y = max_y - min_y
    center_y = (max_y + min_y) / 2

    # вычисление коэффициентов масштабирования
    scale_x = (image_width / 1.1) / diff_x
    scale_y = (image_height / 1.1) / diff_y

    scale = scale_matrix(scale_x, scale_y)

    # вычисление коэффициентов переноса
    tx = image_width / 2 - center_x * scale_x
    ty = image_height / 2 - center_y * scale_y

    translation = translation_matrix(tx, ty)

    transformation = np.dot(translation, scale)

    scaled_vertices = []
    for vertex in vertices:
        point = np.array([vertex[0], vertex[1], 1])
        transformed_point = np.dot(transformation, point)
        scaled_vertices.append([transformed_point[0], transformed_point[1]])

    return scaled_vertices


def draw_line(image, x0, y0, x1, y1, color):
    x0, x1, y0, y1 = map(int, [x0, x1, y0, y1])
    dx, dy = map(abs, [x1 - x0, y1 - y0])
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if x0 < width and y0 < height:
            image[y0, x0, :] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def fill_triangle(image, vertices, color):
    def sign(p1, p2, p3):
        """определение положения точки p3 относительно стороны (p1, p2)
        >0: p3 слева от стороны; <0: p3 справа от стороны; ==0: p3 на одной прямой со стороной"""
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def point_in_triangle(pt, v1, v2, v3):
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        return all(d < 0 for d in [d1, d2, d3]) or all(d > 0 for d in [d1, d2, d3])

    v1, v2, v3 = vertices
    min_x = max(0, int(min(v1[0], v2[0], v3[0])))
    max_x = min(width - 1, int(max(v1[0], v2[0], v3[0])))
    min_y = max(0, int(min(v1[1], v2[1], v3[1])))
    max_y = min(height - 1, int(max(v1[1], v2[1], v3[1])))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_triangle([x, y], v1, v2, v3):
                image[y, x, :] = color


def draw_model(vertices, faces, height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    scaled_vertices = scale_and_translate(vertices, width, height)
    color = [255, 0, 0]

    for i, face in enumerate(faces):
        v1 = scaled_vertices[face[0]]
        v2 = scaled_vertices[face[1]]
        v3 = scaled_vertices[face[2]]

        # отрисовка рёбер
        draw_line(image, v1[0], v1[1], v2[0], v2[1], color)
        draw_line(image, v2[0], v2[1], v3[0], v3[1], color)
        draw_line(image, v3[0], v3[1], v1[0], v1[1], color)

        # заливка каждого второго треугольника
        if i % 2 != 0:
            fill_triangle(image, [v1, v2, v3], color)

    return image


def save_image(image, file_path):
    plt.ylim(height, 0)
    plt.xlim(width, 0)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.imshow(image)
    plt.savefig(file_path)
    plt.show()


if __name__ == "__main__":
    height, width = 1080, 1920
    vertices, faces = read_obj('teapot.obj')
    image = draw_model(vertices, faces, height, width)
    output_file = 'output.png'
    save_image(image, output_file)
