import torch

from functions.svd_operators import Deblurring as BaseDeblurring


class DeblurringColor(BaseDeblurring):
    """
    TMI26 variant of Deblurring that correctly repeats singular values across RGB channels.
    """

    def singulars(self):
        return self._singulars.repeat(self.channels)

    def A(self, vec):
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def At(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def A_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()

        factors = self._safe_inverse(singulars)

        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))

    @staticmethod
    def _safe_inverse(values, eps=1e-12):
        return torch.where(values.abs() > eps, 1.0 / values, torch.zeros_like(values))

    @staticmethod
    def _expanded_singulars(base_singulars, img_dim, device):
        expanded = torch.zeros(img_dim ** 2, device=device, dtype=base_singulars.dtype)
        expanded[:base_singulars.size(0)] = base_singulars
        return expanded

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        temp_vec = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp_vec = self.img_by_mat(temp_vec, self.V_small).reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        singulars = self._expanded_singulars(self._singulars_orig, self.img_dim, vec.device)
        lambda_t = torch.ones(self.img_dim ** 2, device=vec.device, dtype=vec.dtype)
        inverse_singulars = self._safe_inverse(singulars)

        if a != 0 and sigma_y != 0:
            threshold = a * sigma_y * inverse_singulars
            change_index = sigma_t < threshold
            lambda_t = torch.where(
                change_index,
                singulars * sigma_t * (1 - eta ** 2) ** 0.5 / (a * sigma_y),
                lambda_t,
            )

        lambda_t = lambda_t.reshape(1, -1, 1)
        temp_vec = temp_vec * lambda_t

        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device, dtype=vec.dtype)
        temp[:, self._perm, :] = temp_vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1)
        temp_eps = temp_eps[:, :, self._perm].permute(0, 2, 1)

        singulars = self._expanded_singulars(self._singulars_orig, self.img_dim, vec.device)
        inverse_singulars = self._safe_inverse(singulars)

        d1_t = torch.ones(self.img_dim ** 2, device=vec.device, dtype=vec.dtype) * sigma_t * eta
        d2_t = torch.ones(self.img_dim ** 2, device=vec.device, dtype=vec.dtype) * sigma_t * (1 - eta ** 2) ** 0.5

        if a != 0 and sigma_y != 0:
            threshold = a * sigma_y * inverse_singulars

            low_mask = sigma_t < threshold
            d1_t = torch.where(low_mask, sigma_t * eta, d1_t)
            d2_t = torch.where(low_mask, torch.zeros_like(d2_t), d2_t)

            high_mask = sigma_t > threshold
            radicand = sigma_t ** 2 - (a * sigma_y * inverse_singulars) ** 2
            radicand = torch.clamp(radicand, min=0.0)
            d1_t = torch.where(high_mask, torch.sqrt(radicand), d1_t)
            d2_t = torch.where(high_mask, torch.zeros_like(d2_t), d2_t)

            zero_mask = singulars.abs() <= 1e-12
            d1_t = torch.where(zero_mask, sigma_t * eta, d1_t)
            d2_t = torch.where(zero_mask, sigma_t * (1 - eta ** 2) ** 0.5, d2_t)

        d1_t = d1_t.reshape(1, -1, 1)
        d2_t = d2_t.reshape(1, -1, 1)

        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_vec_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device, dtype=vec.dtype)
        temp_vec_new[:, self._perm, :] = temp_vec
        out_vec = self.mat_by_img(self.V_small, temp_vec_new.permute(0, 2, 1))
        out_vec = self.img_by_mat(out_vec, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        temp_eps_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device, dtype=vec.dtype)
        temp_eps_new[:, self._perm, :] = temp_eps
        out_eps = self.mat_by_img(self.V_small, temp_eps_new.permute(0, 2, 1))
        out_eps = self.img_by_mat(out_eps, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        return out_vec + out_eps
