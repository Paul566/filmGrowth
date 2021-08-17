from numba import jit
import numpy as np
import random
from PIL import Image, ImageDraw


@jit(nopython=True)
def num_faces_disloc(h, i, j, n):
    ans = 1
    left = i - 1
    if left == -1:
        left = 1
    right = i + 1
    if right == n:
        right = n - 2
    up = j + 1
    if up == n:
        up = n - 2
    down = j - 1
    if down == -1:
        down = 1

    if h[right][j] >= h[i][j]:
        ans += 1
    if h[left][j] >= h[i][j]:
        ans += 1
    if (j + 1 > n / 2) and (j <= n / 2) and (i >= n / 2):
        if h[i][up] > h[i][j]:
            ans += 1
    else:
        if h[i][up] >= h[i][j]:
            ans += 1
    if (j > n / 2) and (j - 1 <= n / 2) and (i >= n / 2):
        if h[i][down] >= h[i][j] - 1:
            ans += 1
    else:
        if h[i][down] >= h[i][j]:
            ans += 1
    return ans


@jit(nopython=True)
def num_faces(h, i, j, n):
    ans = 1
    left = i - 1
    if (left < 0):
        left = 1
    right = i + 1
    if (right >= n):
        right = n - 2
    up = j + 1
    if (up >= n):
        up = n - 2
    down = j - 1
    if (down < 0):
        down = 1

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
def move(h, i, j, n, exps, disloc=False):
    left = i - 1
    if (left < 0):
        left = 0
    right = i + 1
    if (right >= n):
        right = n - 1
    up = j + 1
    if (up >= n):
        up = n - 1
    down = j - 1
    if (down < 0):
        down = 0

    if disloc and not ((j + 1 > n / 2) and (j <= n / 2) and (i >= n / 2)) \
            and not ((j > n / 2) and (j - 1 <= n / 2) and (i >= n / 2)) and \
            ((h[i][j] == h[right][j]) and (h[i][j] == h[left][j]) and (h[i][j] == h[i][up]) and (
                    h[i][j] == h[i][down])):
        return
    if disloc and (j + 1 > n / 2) and (j <= n / 2) and (i >= n / 2) and (h[i][j] == h[right][j]) and (
            h[i][j] == h[left][j]) and (h[i][j] == h[i][up] + 1) and (h[i][j] == h[i][down]):
        return
    if disloc and (j > n / 2) and (j - 1 <= n / 2) and (i >= n / 2) and (h[i][j] == h[right][j]) and (
            h[i][j] == h[left][j]) and (h[i][j] == h[i][up]) and (h[i][j] - 1 == h[i][down]):
        return

    p_stay = exps[num_faces(h, i, j, n)]
    if disloc:
        p_stay = exps[num_faces_disloc(h, i, j, n)]

    h[i][j] -= 1
    h[left][j] += 1
    p_up = exps[num_faces(h, left, j, n)]
    if disloc:
        p_up = exps[num_faces_disloc(h, left, j, n)]
    if (h[left][j] >= h[i][j] + 2):
        p_up = 0

    h[left][j] -= 1
    h[right][j] += 1
    p_down = exps[num_faces(h, right, j, n)]
    if disloc:
        p_down = exps[num_faces_disloc(h, right, j, n)]
    if (h[right][j] >= h[i][j] + 2):
        p_down = 0

    h[right][j] -= 1
    h[i][down] += 1
    if disloc:
        p_left = exps[num_faces_disloc(h, i, down, n)]
        if ((j > n / 2) and (j - 1 <= n / 2) and (i >= n / 2) and (h[i][down] >= h[i][j] + 1)) or (
                (h[i][down] >= h[i][j] + 2) and not ((j > n / 2) and (j - 1 <= n / 2) and (i >= n / 2))):
            p_left = 0
    else:
        p_left = exps[num_faces(h, i, down, n)]
        if (j == 0) or (h[i][down] >= h[i][j] + 2):
            p_left = 0

    h[i][down] -= 1
    h[i][up] += 1
    if disloc:
        p_right = exps[num_faces_disloc(h, i, up, n)]
        if ((j + 1 > n / 2) and (j <= n / 2) and (i >= n / 2) and (h[i][up] >= h[i][j] + 3)) or (
                (h[i][up] >= h[i][j] + 2) and not (
                (j + 1 > n / 2) and (j <= n / 2) and (i >= n / 2))):
            p_right = 0
    else:
        p_right = exps[num_faces(h, i, up, n)]
        if (h[i][up] >= h[i][j] + 2) or (j == n - 1):
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
def move_all(h, n, exps, dir, disloc=False):
    for i in range(n):
        for j in range(n):
            if dir == 0:
                move(h, i, j, n, exps, disloc)
            if dir == 1:
                move(h, n - 1 - i, n - 1 - j, n, exps, disloc)
            if dir == 2:
                move(h, j, i, n, exps, disloc)
            if dir == 3:
                move(h, n - 1 - j, n - 1 - i, n, exps, disloc)


@jit(nopython=True)
def add_atom(h, n):
    i = random.random()
    j = random.random()
    h[int(np.floor(i * n))][int(np.floor(j * n))] += 1


def image(h, n, width, disloc=False):
    im = Image.new('RGB', (width, width), 'white')
    draw = ImageDraw.Draw(im)
    for i in range(n):
        for j in range(n):
            c = int(255 - h[i][j] * 50)
            c -= int((np.pi + np.angle(-(i + 1) + n / 2 - (j - n / 2) * 1j)) / 2 / np.pi * 50)
            draw.rectangle([(i * width / n, j * width / n), ((i + 1) * width / n, (j + 1) * width / n)], fill=(c, c, c))
    for i in range(n + 1):
        for j in range(n + 1):
            if h[i % n][j % n] != h[(i - 1) % n][j % n]:
                draw.line([(i * width / n, j * width / n), (i * width / n, (j + 1) * width / n)], fill='black', width=3)
            if (j > n / 2) and (j - 1 <= n / 2) and (i >= n / 2) and disloc:
                if h[i % n][j % n] != h[i % n][(j - 1) % n] + 1:
                    draw.line([(i * width / n, j * width / n), ((i + 1) * width / n, j * width / n)], fill='black',
                              width=3)
            else:
                if h[i % n][j % n] != h[i % n][(j - 1) % n]:
                    draw.line([(i * width / n, j * width / n), ((i + 1) * width / n, j * width / n)], fill='black',
                              width=3)
    if disloc:
        draw.ellipse([(width / 2 - 5, width / 2 + width / n - 5), (width / 2 + 5, width / 2 + width / n + 5)],
                     fill='red', )

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
    n = 50
    num_frames = 200
    num_steps = 5000000
    k = 1000  # a new cube is added this many timesteps
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
        move_all(h, n, exps, i % 4, disloc=True)
        if i * num_frames % num_steps == 0:
            images.append(image(h, n, width, disloc=True))
    images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0)


if __name__ == '__main__':
    create()
