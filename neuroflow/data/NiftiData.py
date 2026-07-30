from h5functions import save_to_h5

class NiftiData:
    def __init__(self):
        self.spacing = []

        self.u = None
        self.v = None
        self.w = None

        self.u_mag = None
        self.v_mag = None
        self.w_mag = None

        self.u_venc = None
        self.v_venc = None
        self.w_venc = None

        self.mask = None

    def set_velocity_components(self, u, v, w, mag_u, mag_v, mag_w, venc_u, venc_v, venc_w, spacing, mask=None):
        self.u = u
        self.v = v
        self.w = w

        self.u_mag = mag_u
        self.v_mag = mag_v
        self.w_mag = mag_w

        self.u_venc = venc_u
        self.v_venc = venc_v
        self.w_venc = venc_w

        self.spacing = spacing
        self.mask = mask

    def save_dataset(self, output_filepath, frame_index, save_mask=False):
        assert self.u is not None, "Velocity components are not set"

        save_to_h5(output_filepath, "triggerTimes", float(frame_index))

        save_to_h5(output_filepath, "u", self.u)
        save_to_h5(output_filepath, "v", self.v)
        save_to_h5(output_filepath, "w", self.w)

        save_to_h5(output_filepath, "mag_u", self.u_mag)
        save_to_h5(output_filepath, "mag_v", self.v_mag)
        save_to_h5(output_filepath, "mag_w", self.w_mag)

        save_to_h5(output_filepath, "venc_u", self.u_venc)
        save_to_h5(output_filepath, "venc_v", self.v_venc)
        save_to_h5(output_filepath, "venc_w", self.w_venc)

        save_to_h5(output_filepath, "dx", self.spacing)

        if save_mask and self.mask is not None:
            save_to_h5(output_filepath, "mask", self.mask)
