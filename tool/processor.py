import numpy as np
import cv2


class Processor:
    @staticmethod
    def shake(value, ratio):
        return value * ratio * (np.random.random() - 0.5) * 2

    def drift(self, location, df_ratio):
        x, y, w, h = location
        x += self.shake(w, df_ratio)
        y += self.shake(h, df_ratio)
        return [int(x), int(y), w, h]

    def scale(self, location, sc_ratio):
        x, y, w, h = location
        w += self.shake(w, sc_ratio)
        h += self.shake(h, sc_ratio)
        return [int(x), int(y), w, h]

    @staticmethod
    def rectify(location):
        x, y, w, h = location
        cx = x + w * 0.5
        cy = y + h * 0.5
        w = h = max(w, h)
        x = cx - w * 0.5
        y = cy - h * 0.5
        location = [x, y, w, h]
        location = list(map(int, location))
        return location

    @staticmethod
    def shear(img, location, padding=False):
        if len(img.shape) == 3:
            channel = img.shape[2]
        else:
            channel = 1
        shape = img.shape[:2]
        x, y, w, h = map(int, location)

        if channel == 3:
            tmp = np.zeros((h, w) + (channel,))
        else:
            tmp = np.zeros((h, w))
        if y >= shape[0] or x >= shape[1] or y + h <= 0 or x + w <= 0:
            return tmp
        if padding:
            x_tmp = max(0, -x)
            y_tmp = max(0, -y)
            w_tmp = min(x + w, w, shape[1] - x, shape[1])
            h_tmp = min(y + h, h, shape[0] - y, shape[0])

            x_img = max(0, x)
            y_img = max(0, y)
            w_img = min(x + w, w, shape[1] - x, shape[1])
            h_img = min(y + h, h, shape[0] - y, shape[0])
            tmp[y_tmp:y_tmp + h_tmp, x_tmp:x_tmp + w_tmp] = img[y_img:y_img + h_img, x_img:x_img + w_img]

        else:
            x = max(0, x)
            y = max(0, y)
            tmp = img[y:y + h, x:x + w]

        return tmp

    def contrast(self, img, con_scale=0.2, clip=False):
        mean = img.mean()
        img = (img - mean) * (1 + self.shake(1, con_scale)) + mean
        if clip:
            img = np.clip(img, 0, 255)
        return img

    def brightness(self, img, bri_drift=50, clip=False):
        img = img + self.shake(bri_drift, 1)
        if clip:
            img = np.clip(img, 0, 255)
        return img

    @staticmethod
    def flip(img, p_=0.5):
        if np.random.random() < p_:
            return img[:, ::-1]
        else:
            return img

    def hue(self, img, hue_ratio=0.05, clip=False):
        if len(img.shape) != 3:
            return img
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.shake(180, hue_ratio)) % 180
        hsv = np.clip(hsv, 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.int)
        if clip:
            img = np.clip(img, 0, 255)
        return img

    def saturate(self, img, sat_ratio=0.2, clip=False):
        if len(img.shape) != 3:
            return img
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] *= 1 - sat_ratio * 0.8 + self.shake(sat_ratio, 1)
        hsv = np.clip(hsv, 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.int)
        if clip:
            img = np.clip(img, 0, 255)
        return img

    def hue_saturate(self, img, hue_ratio=0.05, sat_ratio=0.2, clip=False):
        if len(img.shape) != 3:
            return img
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.shake(180, hue_ratio)) % 180
        hsv[:, :, 1] *= 1 - sat_ratio * 0.8 + self.shake(sat_ratio, 1)
        hsv = np.clip(hsv, 0, 255)
        hsv = hsv.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.int)
        if clip:
            img = np.clip(img, 0, 255)
        return img

    @staticmethod
    def noise(img, p_=0.9, noise_rage=10, mask_threshold=None, clip=False):
        if np.random.random() > p_:
            return img
        if mask_threshold is None:
            mask_threshold = np.random.random() * 0.1 + 0.9
        mask = np.random.random(img.shape)
        mask = mask > mask_threshold
        img[mask] += np.random.randint(-noise_rage, noise_rage, img.shape)[mask]
        if clip:
            img = np.clip(img, 0, 255)
        return img

    @staticmethod
    def padding_rectify(img):
        radius = max(img.shape[:2])
        target_shape = (radius, radius) + img.shape[2:]
        target = np.zeros(target_shape).astype(np.uint8)
        x = (radius - img.shape[1]) // 2
        y = (radius - img.shape[0]) // 2
        target[y:img.shape[0] + y, x:img.shape[1] + x] = img
        return target

    def augment(self, img, location=None, target_shape=None, padding=True, drift_ratio=0.02, scale_ratio=0.02,
                rectify=False, contrast=True, brightness=True, flip=True, hue=True, saturate=True, noise=True):
        img = img.astype(np.float)
        if location:
            if scale_ratio:
                location = self.scale(location, scale_ratio)
            if drift_ratio:
                location = self.drift(location, drift_ratio)
            if rectify:
                location = self.rectify(location)
            crop = self.shear(img, location, padding)

        else:
            crop = img
        if contrast:
            crop = self.contrast(crop)
        if brightness:
            crop = self.brightness(crop)
        if flip:
            crop = self.flip(crop)
        if hue and saturate:
            crop = self.hue_saturate(crop)
        else:
            if hue:
                crop = self.hue(crop)
            if saturate:
                crop = self.saturate(crop)
        # if noise:
        #     crop = self.noise(crop)
        crop = self.padding_rectify(crop)
        if target_shape:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
            crop = cv2.resize(crop, target_shape)
        return np.clip(crop, 0, 255).astype(np.uint8)


processor = Processor()

if __name__ == '__main__':
    view = cv2.imread('1275322409.jpg')
    p = Processor()
    while True:
        # t = p.augment(view, [74, 107, 36, 44], (500, 500), True, 0.05, 0.05, False, False, False, False, True, False,
        #               False)
        t = p.augment(view, [74, 107, 36, 44], (500, 500), True)
        cv2.imshow('crop', t)
        cv2.waitKey()
