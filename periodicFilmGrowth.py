from numba import jit
import numpy as np
import random
from PIL import Image, ImageDraw


@jit(nopython=True)
def num_faces(h, i, j, n):
    ans = 1
    left = i - 1
    if (left < 0):
        left = n - 1
    right = i + 1
    if (right >= n):
        right = 0
    up = j + 1
    if (up >= n):
        up = 0
    down = j - 1
    if (down < 0):
        down = n - 1
    if h[right][j] >= h[i][j]:
        ans += 1
    if h[left][j] >= h[i][j]:
        ans += 1
    if h[i][up] >= h[i][j]:
        ans += 1
    if h[i][down] >= h[i][j]:
        ans += 1
    return ans


@jit(nopython=True)
def move(h, i, j, n, exps):
    left = i - 1
    if (left < 0):
        left = n - 1
    right = i + 1
    if (right >= n):
        right = 0
    up = j + 1
    if (up >= n):
        up = 0
    down = j - 1
    if (down < 0):
        down = n - 1

    if (h[i][j] == h[right][j]) and (h[i][j] == h[left][j]) and (h[i][j] == h[i][up]) and (h[i][j] == h[i][down]):
        return

    p_stay = exps[num_faces(h, i, j, n)]
    h[i][j] -= 1
    h[left][j] += 1
    p_up = exps[num_faces(h, left, j, n)]
    if h[left][j] >= h[i][j] + 2:
        p_up = 0
    h[left][j] -= 1
    h[right][j] += 1
    p_down = exps[num_faces(h, right, j, n)]
    if h[right][j] >= h[i][j] + 2:
        p_down = 0
    h[right][j] -= 1
    h[i][down] += 1
    p_left = exps[num_faces(h, i, down, n)]
    if h[i][down] >= h[i][j] + 2:
        p_left = 0
    h[i][down] -= 1
    h[i][up] += 1
    p_right = exps[num_faces(h, i, up, n)]
    if h[i][up] >= h[i][j] + 2:
        p_right = 0
    h[i][up] -= 1
    h[i][j] += 1

    full_prob = p_stay + p_left + p_right + p_down + p_up
    p_stay /= full_prob
    p_left /= full_prob
    p_right /= full_prob
    p_up /= full_prob
    p_down /= full_prob

    r = random.random()
    if r < p_up:
        h[i][j] -= 1
        h[left][j] += 1
        return
    if r < p_up + p_down:
        h[i][j] -= 1
        h[right][j] += 1
        return
    if r < p_up + p_down + p_left:
        h[i][j] -= 1
        h[i][down] += 1
        return
    if r < p_up + p_down + p_left + p_right:
        h[i][j] -= 1
        h[i][up] += 1
        return


@jit(nopython=True)
def move_all(h, n, exps, dir):
    for i in range(n):
        for j in range(n):
            if dir == 0:
                move(h, i, j, n, exps)
            if dir == 1:
                move(h, n - 1 - i, n - 1 - j, n, exps)
            if dir == 2:
                move(h, j, i, n, exps)
            if dir == 3:
                move(h, n - 1 - j, n - 1 - i, n, exps)


@jit(nopython=True)
def add_atom(h, n):
    i = random.random()
    j = random.random()
    h[int(np.floor(i * n))][int(np.floor(j * n))] += 1


def image(h, n, width):
    im = Image.new('RGB', (width, width), 'white')
    draw = ImageDraw.Draw(im)
    for i in range(n):
        for j in range(n):
            c = int(255 - h[i][j] * 50)
            draw.rectangle([(i * width / n, j * width / n), ((i + 1) * width / n, (j + 1) * width / n)], fill=(c, c, c))
    for i in range(n + 1):
        for j in range(n + 1):
            if h[i % n][j % n] != h[(i - 1) % n][j % n]:
                draw.line([(i * width / n, j * width / n), (i * width / n, (j + 1) * width / n)], fill='black', width=3)
            if h[i % n][j % n] != h[i % n][(j - 1) % n]:
                draw.line([(i * width / n, j * width / n), ((i + 1) * width / n, j * width / n)], fill='black', width=3)

    return im


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def create():
    n = 100
    num_frames = 200
    num_steps = 200000
    k = 10  # a new cube is added this many timesteps
    h = np.zeros((n, n))
    kT = 0.3
    exps = np.zeros(6)
    for i in range(6):
        exps[i] = np.exp(i / kT)
    width = 600
    images = []

    for i in range(num_steps):
        printProgressBar(i + 1, num_steps, prefix='Progress:', suffix='Complete', length=50)
        if i % k == 0:
            add_atom(h, n)
        move_all(h, n, exps, i % 4)
        if i * num_frames % num_steps == 0:
            images.append(image(h, n, width))
    images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0)


if __name__ == '__main__':
    create()
