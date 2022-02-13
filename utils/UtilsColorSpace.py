"""Functions for converting tensors between color spaces, slightly modified from
https://github.com/niopeng/CAM-Net/blob/main/code/utils/util.py
"""
import torch
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

color_output_mode = "A*L*B*L"  # "A*L*B*L" or "AB"
AB_range = "standard"  # "standard" or "real"
# AB_range = "real"
l_center = 0.0
l_norm = 100.0

if AB_range == "standard":
    a_center = 0.0
    a_norm = 127.0
    b_center = 0.0
    b_norm = 127.0
elif AB_range == "real":
    a_center = 6.0345
    a_norm = 92.2195
    b_center = - 6.6905
    b_norm = 101.1725




def rgb2xyz(rgb):
    mask = (rgb > .04045).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    return out


def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (1.055 * ((rgb + 1e-12) ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)

    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if xyz_scale.is_cuda:
        mask = mask.cuda()

    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)

    if z_int.is_cuda:
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if out.is_cuda:
        mask = mask.cuda()

    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc
    return out

def lab2rgb(lab_tensor):
    """Returns LAB BxCxHxW tensor [lab_tensor] in the RGB color space."""
    l_tmp = lab_tensor[:, [0], :, :]
    a_tmp = lab_tensor[:, [1], :, :]
    b_tmp = lab_tensor[:, [2], :, :]

    # way 1: max(x, eps)
    # way 2: min(x + eps, 1)
    # way 3: (x + eps) / (1 + eps)
    l_tmp_d = l_tmp
    # if color_output_mode == "A*L*B*L":
    #     l_tmp_d = torch.max(torch.Tensor((0.01,)).to(device), l_tmp)
    #     a_tmp /= l_tmp_d
    #     b_tmp /= l_tmp_d
    l_tmp = l_tmp_d * l_norm + l_center
    a_tmp = a_tmp * a_norm + a_center
    b_tmp = b_tmp * b_norm + b_center

    lab_tmp = torch.cat((l_tmp, a_tmp, b_tmp), dim=1)

    # print("LAB TENSOR IN CONVERT TO RGB", lab_tmp, "\n\n")

    result = xyz2rgb(lab2xyz(lab_tmp))

    # print("RGB TENSOR IN CONVERT TO RGB", result, "\n\n")

    return result

def rgb2lab(rgb_tensor):
    """Returns RGB BxCxHxW tensor [rgb_tensor] in the LAB color space."""
    # print("RGB TENSOR IN CONVERT TO LAB", rgb_tensor, "\n\n")
    result = xyz2lab(rgb2xyz(rgb_tensor))

    result[:, 0] /= 100
    result[:, 1:] /= 127
    result[:, 2:] /= 127

    # print("LAB TENSOR IN CONVERT TO LAB", result, "\n\n")

    return result

def rgb2lab_with_dims(input):
    if isinstance(input, list):
        return [rgb2lab_with_dims(x) for x in input]
    elif isinstance(input, torch.Tensor) and len(input.shape) == 4:
        return rgb2lab(input.to(device))
    elif isinstance(input, torch.Tensor) and len(input.shape) == 3:
        return rgb2lab(input.to(device).unsqueeze(0)).squeeze(0)

def lab2rgb_with_dims(input):
    if isinstance(input, list):
        return [lab2rgb_with_dims(x) for x in input]
    elif isinstance(input, torch.Tensor) and len(input.shape) == 4:
        return lab2rgb(input.to(device))
    elif isinstance(input, torch.Tensor) and len(input.shape) == 3:
        return lab2rgb(input.to(device).unsqueeze(0)).squeeze(0)
