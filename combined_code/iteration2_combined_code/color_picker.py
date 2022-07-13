
HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([159, 50, 70]),
            'upper': np.array([179, 255, 255])
        },
        {
            'lower': np.array([0, 50, 70]),
            'upper': np.array([9, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([15, 0, 0]),
            'upper': np.array([36, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([36, 0, 0]),
            'upper': np.array([86, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([100, 150, 0]),
            'upper': np.array([140, 255, 255])
        }
    ],
    'orange': [
        {
            'lower': np.array([18, 40, 90]),
            'upper': np.array([27, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 25% of V
    'black': [
        {
            'lower': np.array([97, 39, 0]),
            'upper': np.array([179, 255, 30])
        }
    ],
    # gray is all H values, lower 15% of S, & between 26-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([179, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([180, 38, 255])
        }
    ]
}


def create_mask(img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # noinspection PyUnresolvedReferences
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            # noinspection PyUnresolvedReferences
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )
    return cv2.countNonZero(mask)  # np.count_nonzero(mask==1)


def color_picker(img):
    masks = []
    colors = {"0": "red", '1': "blue", '2': "yellow",
              '3': "green", '4': "orange", '5': "black", '6': "white"}
    masks.append(('0', create_mask(img, ['red'])))
    masks.append(('1', create_mask(img, ['blue'])))
    # masks.append(('2', create_mask(img, ['yellow'])))
    masks.append(('3', create_mask(img, ['green'])))
    masks.append(('4', create_mask(img, ['orange'])))
    masks.append(('5', create_mask(img, ['black'])))
    masks.append(('6', create_mask(img, ['white'])))
    masks.sort(key=lambda x: x[1], reverse=True)
    return colors[masks[0][0]]
