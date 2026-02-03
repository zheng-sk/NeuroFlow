import torch
import torch.nn.functional as F


def create_divergence_kernels(device, dtype):
    """
    Create kernels in 3 different directions to calculate central differences.
    """
    filter_x = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    filter_x[0, 0, 0, 1, 1] = 1.0
    filter_x[0, 0, 2, 1, 1] = -1.0

    filter_y = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    filter_y[0, 0, 1, 0, 1] = 1.0
    filter_y[0, 0, 1, 2, 1] = -1.0

    filter_z = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    filter_z[0, 0, 1, 1, 0] = 1.0
    filter_z[0, 0, 1, 1, 2] = -1.0

    return filter_x, filter_y, filter_z


def calculate_gradient(image, kernel):
    """
    Calculate the gradient (edge) of an image using a predetermined kernel.
    Expects image shape [B, D, H, W].
    """
    image = image.unsqueeze(1)
    image = F.pad(image, (1, 1, 1, 1, 1, 1), mode="replicate")
    conv = F.conv3d(image, kernel)
    return conv.squeeze(1)


def calculate_divergence(u, v, w):
    """
    Calculate divergence for the corresponding velocity component.
    """
    kernels = create_divergence_kernels(device=u.device, dtype=u.dtype)
    dudx = calculate_gradient(u, kernels[0])
    dvdy = calculate_gradient(v, kernels[1])
    dwdz = calculate_gradient(w, kernels[2])
    return dudx, dvdy, dwdz


def calculate_divergence_loss2(u, v, w, u_pred, v_pred, w_pred):
    divpx, divpy, divpz = calculate_divergence(u_pred, v_pred, w_pred)
    divx, divy, divz = calculate_divergence(u, v, w)
    return (divpx - divx) ** 2 + (divpy - divy) ** 2 + (divpz - divz) ** 2


def calculate_relative_error(u_pred, v_pred, w_pred, u_hi, v_hi, w_hi, binary_mask):
    # if epsilon is set to 0, we will get nan and inf
    epsilon = 1e-5

    u_diff = (u_pred - u_hi) ** 2
    v_diff = (v_pred - v_hi) ** 2
    w_diff = (w_pred - w_hi) ** 2

    diff_speed = torch.sqrt(u_diff + v_diff + w_diff)
    actual_speed = torch.sqrt(u_hi**2 + v_hi**2 + w_hi**2)

    # actual speed can be 0, resulting in inf
    relative_speed_loss = diff_speed / (actual_speed + epsilon)

    # Make sure the range is between 0 and 1
    relative_speed_loss = torch.clamp(relative_speed_loss, 0.0, 1.0)

    # Apply correction, only use the diff speed if actual speed is zero
    corrected_speed_loss = torch.where(actual_speed != 0, relative_speed_loss, diff_speed)

    multiplier = 1e4  # round it so we don't get any infinitesimal number
    corrected_speed_loss = torch.round(corrected_speed_loss * multiplier) / multiplier

    # Apply mask
    corrected_speed_loss = torch.where(binary_mask == 1.0, corrected_speed_loss, torch.zeros_like(corrected_speed_loss))

    # Calculate the mean from the total non zero accuracy, divided by the masked area
    mean_err = corrected_speed_loss.sum(dim=(1, 2, 3)) / (binary_mask.sum(dim=(1, 2, 3)) + 1)
    return mean_err * 100
