from torch import nn

def pad_tensor(input,divide = 16):#divide 为下采样倍数
    depth_org,height_org, width_org = input.shape[2],input.shape[3], input.shape[4]

    if depth_org % divide != 0 or width_org % divide != 0 or height_org % divide != 0:
        depth_res = depth_org % divide
        width_res = width_org % divide
        height_res = height_org % divide
        if depth_res != 0:
            depth_div = divide - depth_res
            pad_front = int(depth_div / 2)
            pad_back = int(depth_div - pad_front)
        else:
            pad_front = 0
            pad_back = 0

        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad3d((pad_left, pad_right, pad_top, pad_bottom,pad_front,pad_back))
        input = padding(input)
    else:
        pad_front = 0
        pad_back = 0
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    depth, height, width = input.data.shape[2], input.data.shape[3], input.data.shape[4]
    assert depth % divide == 0, 'depth cant divided by stride'
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom):
    depth, height, width = input.data.shape[2], input.data.shape[3], input.data.shape[4]
    return input[:, :, pad_front: depth - pad_back, pad_top: height - pad_bottom, pad_left: width - pad_right]
