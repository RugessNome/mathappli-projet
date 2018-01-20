
import mnist 
from mnist import show

import numpy as np

training_data = mnist.load(mnist.MNIST_TRAINING_DATA, mnist.MNIST_FORMAT_LIST_OF_PAIR)[0:10000]
test_data = mnist.load(mnist.MNIST_TEST_DATA, mnist.MNIST_FORMAT_LIST_OF_PAIR)

# Image binarization

def binarize(img, treshold = 200):
    w = len(img)
    h = len(img[0])
    ret = img.copy()
    for i in range(w):
        for j in range(h):
            ret[i][j] = 1 if img[i][j] >= treshold else 0
    return ret



# Loop detection

def fillArea(img, x, y, color = 1):
    if img[x][y] != 0:
        return
    img[x][y] = color
    fillArea(img, max(0, x-1), y, color)
    fillArea(img, min(len(img)-1, x+1), y, color)
    fillArea(img, x, max(0, y-1), color)
    fillArea(img, x, min(len(img[0])-1, y+1), color)


def detectLoops(binimg, colors = [240, 210, 180]):
    """
    Takes a binarized image and returns an image with the 
    loops filled with the given ``colors```and the number of 
    loops detected. (i.e returns (img, nbLoops))
    """
    img = binimg.copy()
    fillArea(img, 0, 0)
    c = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 0:
                fillArea(img, i, j, colors[c % len(colors)])
                c += 1
    return img, c

def fill_loops(inimg):
    """
    Returns an image with the loops filled with ink.
    This uses the loop dectection algorithm as implemented in 
    ```detectLoops``.
    """
    loopimg = inimg.copy()
    bgcolor = 1
    loopcolor = 2
    fillArea(loopimg, 0, 0, bgcolor)
    c = 0
    for i in range(len(loopimg)):
        for j in range(len(loopimg[0])):
            if loopimg[i][j] == 0:
                fillArea(loopimg, i, j, loopcolor)
    out = inimg.copy()
    for i in range(len(loopimg)):
        for j in range(len(loopimg[0])):
            if loopimg[i][j] == loopcolor:
                out[i][j] = 255
    return out


# Zoning

def zoning(img, pw = 4, ph = 4):
    ret = []
    for y in range(0, 28, ph):
        for x in range(0, 28, pw):
            zone = 0
            for i in range(pw):
                for j in range(ph):
                    zone += img[y+j][x+i]
            zone = zone / (ph*pw)
            ret.append(zone)
    return ret

def plot_zoning(img, pw = 4, ph = 4):
    zones = zoning(img, pw, ph)
    zoneimg = np.zeros((28,28))
    for y in range(0, 28, ph):
        for x in range(0, 28, pw):
            for i in range(pw):
                for j in range(ph):
                    zoneimg[y+j][x+i] = zones[int(y/ph) * int(28/pw) + int(x/pw)]
    show(zoneimg)


   
# Crossings

def colCrossings(img, col):
    n = 0
    c = 0
    x = col
    for y in range(len(img)):
        if img[y][x] != c:
            n += 1
            c = 1-c
    return n
    
def rowCrossings(img, row):
    n = 0
    c = 0
    y = row
    for x in range(len(img[0])):
        if img[y][x] != c:
            n += 1
            c = 1-c
    return n
    
def rowCrossingsList(img):
    ret = []
    for r in range(len(img)):
        ret.append(rowCrossings(img, r))
    return ret
    
def example_row_crossings(img):
    show(binarize(img))
    rcl = rowCrossingsList(binarize(img))
    print(rcl)
    return rcl
    
    
# Histograms

def hvhistogram(img, xmin = 0, xmax = None, ymin = 0, ymax = None):
    h = len(img)
    w = len(img[0])
    if xmax == None:
        xmax = w
    if ymax == None:
        ymax = h
    vertical = [0] * (xmax-xmin)
    horizontal = [0] * (ymax-ymin)
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if img[y][x] > 0:
               vertical[y] += 1
               horizontal[x] += 1
    return horizontal, vertical

def saveHistogram(image, hh, vh, file, w = 300, h = 200):
    fig, axs = pyplot.subplots(2, 2, sharex=True, sharey=True)
    #fig.set_size_inches(w/100, h/100)
    imgplot = axs[0, 1].imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    axs[0, 1].xaxis.set_ticks_position('top')
    axs[0, 1].yaxis.set_ticks_position('left')
    axs[0, 0].barh(range(0, len(image)), vh)
    axs[1, 1].bar(range(0, len(image)), hh)
    axs[1, 0].axis('off')
    pyplot.savefig(file, dpi = 100)
    pyplot.close(fig)
    

def example_histo(img):
    image = binarize(img)
    hh, vh = hvhistogram(image)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    #fig.set_size_inches(w/100, h/100)
    imgplot = axs[0, 1].imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    axs[0, 1].xaxis.set_ticks_position('top')
    axs[0, 1].yaxis.set_ticks_position('left')
    axs[0, 0].barh(range(0, len(image)), vh)
    axs[1, 1].bar(range(0, len(image)), hh)
    axs[1, 0].axis('off')
    plt.show()
    
# Moments

def moment(img, p, q):
    m = 0
    for x in range(len(img)):
        xp = x**p
        for y in range(len(img[0])):
            m = m + xp * y**q * img[y][x]
    return m
    
def cmoment(img, p, q, xbar = None, ybar = None):
    m = 0
    if xbar == None or ybar == None:
        m00 = moment(img, 0, 0)
        xbar = moment(img, 1, 0) / m00
        ybar = moment(img, 0, 1) / m00
    for x in range(len(img)):
        xp = (x-xbar)**p
        for y in range(len(img[0])):
            yq = (y - ybar)**q
            m = m + xp * yq * img[y][x]
    return m

   
def nbBlackPixels(img):
    return moment(img, 0, 0)
    


# FFT

## This part implements Fourier transform of the image
## using scipy.

import scipy.fftpack as fp

## Functions to go from image to frequency-image and back
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)
freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1), axis=0)


def fft_test(index = 62, harmo = 8):
    label, img = training_data[index]
    
    show(img)
    
    img = img / 255
    
    fft = im2freq(img)
    
    for i in range(5):
        for j in range(5):
            print("fft({},{}) = {}".format(i, j, fft[i][j]))
    
    for i in range(harmo, 28):
        for j in range(harmo, 28):
            fft[i][j] = 0
    
    imgback = freq2im(fft)
    show(imgback * 255)

def fft_figure():
    from matplotlib import pyplot
    import matplotlib as mpl
    fig, axs = pyplot.subplots(3, 3)
    #fig.set_size_inches(w/100, h/100)
    for i in range(3):
        for j in range(3):
            fft = np.zeros((28,28))
            fft[i][j] = 1
            imgplot = axs[i, j].imshow(freq2im(fft), cmap=mpl.cm.Greys)
            imgplot.set_interpolation('nearest')
            axs[i, j].axis('off')
    axs[0, 0].axis('on')
    axs[0, 0].xaxis.set_ticks_position('top')
    axs[0, 0].yaxis.set_ticks_position('left')
    pyplot.show()




# Fourier Transform of contour

def extract_contour(img):
    h, w = len(img), len(img[0])
    directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

    def find_starting_point():
        for y in range(h):
            for x in range(w):
                if img[y][x] > 0:
                    return (x-1,y)
        raise RuntimeError('Could not compute contour of image (image is empty)')

    spt = find_starting_point()

    def pixelcolor(x, y):
        if 0 <= x < w and 0 <= y < h:
            return img[y][x]
        return 0

    def advance(x, y, d):
        def can_go(x, y, d):
            if pixelcolor(x+directions[d][0], y+directions[d][1]) > 0:
                return False 
            if d % 2 == 1: # we need to make further verifications (diagonal movement)
                if pixelcolor(x+directions[(d+7)%8][0], y+directions[(d+7)%8][1]) == 0:
                    return True
                elif pixelcolor(x+directions[(d+1)%8][0], y+directions[(d+1)%8][1]) == 0:
                    return True
                else:
                    return False
            return True

        def can_go_right(x, y, d):
            return can_go(x, y, (d+1)%8)

        dx, dy = directions[d]
        if can_go(x, y, d):
            while pixelcolor(x+dx, y+dy) == 0:
                d = (d+1) % 8
                dx, dy = directions[d]
            return (d+8-1) % 8
        elif can_go_right(x, y, d):
            d = (d+1) % 8
            dx, dy = directions[d]
            while pixelcolor(x+dx, y+dy) == 0:
                d = (d+1) % 8
                dx, dy = directions[d]
            return (d+8-1) % 8
        else:
           while not can_go(x, y, d):
                d = (d+7) % 8
           return d


    visited = -1 * np.ones((h,w))
    def mark_as_visited(x, y):
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        visited[y][x] = len(chain)
    def already_visited(x, y):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return visited[y][x] != -1
    def simplify_chain(ch, x, y):
        assert(already_visited(x,y))
        step = int(visited[y][x])
        return ch[0:step+1]
   
    chain = []
    x, y = spt
    d = advance(x, y, 7)
    chain.append(d)
    dx, dy = directions[d]
    x = x+dx
    y = y+dy
    iter = 0
    while (x,y) != spt:
        d = advance(x, y, d)
        if already_visited(x, y):
            chain = simplify_chain(chain, x, y)
        else:
            mark_as_visited(x, y)
        chain.append(d)
        dx, dy = directions[d]
        x = x+dx
        y = y+dy
        iter += 1
        if iter > h * w: # Obviously we loop infinitely
            raise RuntimeError('Could not compute contour of image')
    return spt, chain



def freeman_chain_img(pt, chain, w = 28, h = 28):
    directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    img = np.zeros((h,w))
    x,y = pt
    img[y][x] = 1
    for c in chain:
        dx, dy = directions[c]
        x += dx
        y += dy
        img[y][x] = 1
    return img

## Adapted from https://3010tangents.wordpress.com/2015/05/12/elliptic-fourier-descriptors/

def traversal_dist(chain):
    x, y = 0, 0
    directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

    if not isinstance(chain, (list)):
        return directions[chain]

    result = []
    for c in chain:
        dx, dy = directions[c]
        x += dx
        y += dy
        result.append((x,y))
    return result

def traversal_time(chain):
    times = [1.0, np.sqrt(2.), 1.0, np.sqrt(2.), 1.0, np.sqrt(2.), 1.0, np.sqrt(2.)]

    if not isinstance(chain, (list)):
        return times[chain]

    t = 0
    result = []
    for c in chain:
        t += times[c]
        result.append(t)
    return result


def calc_dc_components(chain, times = None, dists = None):
    k = len(chain)

    if times == None:
        times = traversal_time(chain)
    if dists == None:
        dists = traversal_dist(chain)

    T = times[-1]

    sum_a0 = 0
    sum_c0 = 0

    for p in range(k):
        dx, dy = traversal_dist(chain[p])
        dt = traversal_time(chain[p]);

        zeta = 0
        delta = 0
        if p > 0:
            zeta = dists[p-1][0] - dx / dt * times[p-1]
            delta = dists[p-1][1] - dy / dt * times[p-1]

        if p > 0:
            sum_a0 += dx / (2*dt) * ((times[p])**2 - (times[p-1])**2) + zeta * (times[p] - times[p-1])
            sum_c0 += dy / (2*dt) * ((times[p])**2 - (times[p-1])**2) + delta * (times[p] - times[p-1])
        else:
            sum_a0 += dx / (2*dt) * (times[p])**2 + zeta * times[p]
            sum_c0 += dy / (2*dt) * (times[p])**2 + delta * times[p]

    return sum_a0 / T, sum_c0 / T

def calc_harmonic_coefficients(chain, n, times = None, dists = None):
    k = len(chain)

    if times == None:
        times = traversal_time(chain)
    if dists == None:
        dists = traversal_dist(chain)

    T = times[-1]

    ntwopi = 2 * n * np.pi
    a, b, c, d, = 0, 0, 0, 0

    for p in range(k):
        tp_prev = 0
        if p > 1:
            tp_prev = times[p-1]
        
        dx, dy = traversal_dist(chain[p])
        dt = traversal_time(chain[p]);
        
        q_x = dx / dt;
        q_y = dy / dt;
        
        a += q_x * (np.cos(ntwopi * times[p] / T) - np.cos(ntwopi * tp_prev / T));
        b += q_x * (np.sin(ntwopi * times[p] / T) - np.sin(ntwopi * tp_prev / T));
        c += q_y * (np.cos(ntwopi * times[p] / T) - np.cos(ntwopi * tp_prev / T));
        d += q_y * (np.sin(ntwopi * times[p] / T) - np.sin(ntwopi * tp_prev / T));   

    r = T / (2 * n**2 * np.pi**2)
    return r * np.array([a, b, c, d])


def fourier_approx(chain, n, m, normalized = True):
    times = traversal_time(chain)
    dists = traversal_dist(chain)
    a, b, c, d = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
         ai, bi, ci, di = calc_harmonic_coefficients(chain, i+1, times, dists);
         a[i] = ai; b[i] = bi; c[i] = ci; d[i] = di

    A0, C0 = calc_dc_components(chain, times, dists)

    if normalized:
        A0 = 0
        C0 = 0

        theta1 = 0.5 * np.arctan(2 * (a[0] * b[0] + c[0] * d[0]) / (a[0]**2 + c[0]**2 - b[0]**2 - d[0]**2))
       
        costh1 = np.cos(theta1);
        sinth1 = np.sin(theta1);
             	 
        a_star_1 = costh1 * a[0] + sinth1 * b[0]
        b_star_1 = -sinth1 * a[0] + costh1 * b[0]
        c_star_1 = costh1 * c[0] + sinth1 * d[0]
        d_star_1 = -sinth1 * c[0] + costh1 * d[0]
       
        psi1 = np.arctan(c_star_1 / a_star_1) ;
        
        E = np.sqrt(a_star_1**2 + c_star_1**2);
        
        cospsi1 = np.cos(psi1);
        sinpsi1 = np.sin(psi1);
        
        for i in range(n):    
            M1 = np.array([[cospsi1, sinpsi1], [-sinpsi1, cospsi1]])
            M2 = np.array([[a[i], b[i]], [c[i], d[i]]])
            M3 = np.array([[np.cos(theta1 * (i+1)), -np.sin(theta1 * (i+1))], [np.sin(theta1 * (i+1)), np.cos(theta1 * (i+1))]])
            normalized = np.dot(np.dot(M1, M2), M3)
            a[i] = normalized[0][0] / E
            b[i] = normalized[0][1] / E
            c[i] = normalized[1][0] / E
            d[i] = normalized[1][1] / E

    pts = []
    for t in range(1, m+1):
        x = 0
        y = 0

        for i in range(n):
            x += (a[i] * np.cos(2 * (i+1) * np.pi * t / m) + b[i] * np.sin(2 * (i+1) * np.pi * t / m))
            y += (c[i] * np.cos(2 * (i+1) * np.pi * t / m) + d[i] * np.sin(2 * (i+1) * np.pi * t / m))

        pts.append((A0+x,C0+y))

    return pts



def plot_fourier_approx(img, n, m, normalized = False):
    spt, chain = extract_contour(img)
    result = fourier_approx(chain, n, m, False)
    X = []
    Y = []
    for x, y in result:
        X.append(x)
        Y.append(y)
    import matplotlib.pyplot as plt
    plt.plot(X, Y)
    plt.show()

def plot_contour(img):
    spt, chain = extract_contour(img)
    show(freeman_chain_img(spt, chain))

def test_extract_contour(data = training_data):
    step = int(len(data) / 100)
    for i in range(len(data)):
        label, img = data[i]
        try:
            spt, chain = extract_contour(img)
        except:
            print('Could not compute contour of image ', i) 
        if i % step == 0:
            print("Progress : {}/{} ({} %)".format(i, len(data), i / step))

def example_contour():
    img, label = training_data[524]
    #img, label = training_data[12]
    #img, label = training_data[2837]
    #img, label = training_data[1418]
    show(img)
    plot_contour(img)
    plot_fourier_approx(img, 8, 100, True)
