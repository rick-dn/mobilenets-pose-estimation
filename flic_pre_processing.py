import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import misc
from tqdm import tqdm
import tensorflow as tf

from flags import FLAGS


class LSPPreProcess:
    __count = 0
    __index = 0

    def __init__(self, is_train=True):

        self.images_dir = FLAGS.images_dir
        self.annotations = loadmat(FLAGS.annotations)
        self.is_train = is_train

    @staticmethod
    def apply_image_norm(image):

        image = image.astype(np.float)
        image = image * (2.0 / 255.0) - 1.0

        return image

    @staticmethod
    def apply_coord_norm(image, joints):
        """Coordinate normalization.
        Subtract coordinate of image center
        and divide by image height and width
        Args:
          image: Resized image
          joints: Resized joints
        Returns:
          joints: Normalized joints
        """

        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        joints -= np.array([center_x, center_y])
        joints[:, 0] /= w
        joints[:, 1] /= h
        return joints

    @staticmethod
    def calc_joints_bbox(joints_gt, image):

        x_min = np.min(joints_gt[:, 0][joints_gt[:, 0] != 0])
        y_min = np.min(joints_gt[:, 1][joints_gt[:, 0] != 0])
        x_max = np.max(joints_gt[:, 0][joints_gt[:, 0] != 0])
        y_max = np.max(joints_gt[:, 1][joints_gt[:, 0] != 0])

        orig_bbox = [x_min, y_min, x_max, y_max]

        x_min -= 50
        y_min -= 50
        x_max += 50
        y_max += 50

        bbox = [x_min.clip(min=0).astype(int),
                y_min.clip(min=0).astype(int),
                x_max.clip(max=image.shape[1]).astype(int),
                y_max.clip(max=image.shape[0]).astype(int)]

        return bbox, orig_bbox

    @staticmethod
    def data_aug(image, joints_gt):

        angle = np.random.randint(-30, 30)
        scale = np.random.uniform(.75, 1.25)

        x_min = np.min(joints_gt[:, 0][joints_gt[:, 0] != 0])
        y_min = np.min(joints_gt[:, 1][joints_gt[:, 0] != 0])
        x_max = np.max(joints_gt[:, 0][joints_gt[:, 0] != 0])
        y_max = np.max(joints_gt[:, 1][joints_gt[:, 0] != 0])

        m_inv = cv.getRotationMatrix2D(((x_min + x_max) // 2, (y_max + y_min) // 2), angle, scale)

        ones = np.ones(shape=(len(joints_gt), 1))
        points_ones = np.hstack([joints_gt, ones])
        joints_gt_aug = m_inv.dot(points_ones.T).T

        if np.min(joints_gt_aug[joints_gt_aug != 0]) - 20 < 1 or \
                np.max(joints_gt_aug[joints_gt_aug != 0] + 20) > 224:

            return image, joints_gt

        image = cv.warpAffine(image, m_inv, (image.shape[1], image.shape[0]))

        return image, joints_gt_aug

    @staticmethod
    def on_the_fly(image, joints_gt, is_train):

        if is_train:
            image, joints_gt = LSPPreProcess.data_aug(image, joints_gt)

        # joints_gt = LSPPreProcess.joints_gt_dense(joints_gt)
        # r_theta = joints_gt[[1, 15], :]

        _, joints_gt_hm = LSPPreProcess.resize_img(image, joints_gt, FLAGS.heatmap_size)
        heat_map = LSPPreProcess.generate_hm(FLAGS.heatmap_size, joints_gt_hm, FLAGS.heatmap_size)
        # joints_gt = LSPPreProcess.get_joint_from_hm(heat_map)
        # joints_gt = LSPPreProcess.resize_joints(joints_gt, 224, 56)

        image = LSPPreProcess.apply_image_norm(image)
        # joints_gt = LSPPreProcess.apply_coord_norm(image, joints_gt)
        # joints_gt[[1, 15], :] = r_theta

        image = np.asarray(image, dtype=np.float32)
        joints_gt = np.asarray(heat_map, dtype=np.float32)

        return image, joints_gt

    @staticmethod
    def calc_r_theta(joints_gt_arm):

        delta_y1 = joints_gt_arm[1, 1] - joints_gt_arm[0, 1]
        delta_x1 = joints_gt_arm[1, 0] - joints_gt_arm[0, 0]
        delta_y2 = joints_gt_arm[2, 1] - joints_gt_arm[1, 1]
        delta_x2 = joints_gt_arm[2, 0] - joints_gt_arm[1, 0]

        theta1 = np.arctan2(delta_y1, delta_x1)
        theta2 = np.arctan2(delta_y2, delta_x2)

        theta = np.abs((np.pi - np.abs(theta1 - theta2))) * (180 / np.pi)
        r = np.sqrt(delta_y1 ** 2 + delta_x1 ** 2) + np.sqrt(delta_y2 ** 2 + delta_x2 ** 2)

        r = r / FLAGS.resize_input_image
        theta = theta / 360

        return [r, theta]

    @staticmethod
    def joints_gt_dense(joints_gt):

        joints_gt_op = np.zeros((FLAGS.no_of_joints + FLAGS.no_of_dense_joints,
                                 FLAGS.coord_per_joint))
        joints_gt_op[[0, 7, 13, 14, 21, 27, 28, 29, 30, 31, 32], :] = joints_gt

        for joint_n, k in zip((0, 1, 3, 4), (1, 7, 15, 21)):
            x, y = (joints_gt[joint_n + 1, :] - joints_gt[joint_n, :]) / 6
            for i in range(5):
                joints_gt_op[k + i + 1, :] = joints_gt[joint_n, :] + [(i + 1) * x, (i + 1) * y]

        joints_gt_op[1, :] = LSPPreProcess.calc_r_theta(joints_gt_op[[0, 7, 13]])
        joints_gt_op[15, :] = LSPPreProcess.calc_r_theta(joints_gt_op[[14, 21, 27]])

        return joints_gt_op

    @staticmethod
    def crop_image(image, joints_gt, bbox):

        x_min, y_min, x_max, y_max = bbox

        image = image[y_min:y_max, x_min:x_max]
        joints_gt -= np.array([x_min, y_min])

        return image, joints_gt

    @staticmethod
    def coord_de_normalize(joints_gt):

        h = FLAGS.resize_input_image
        w = FLAGS.resize_input_image
        center_x, center_y = w // 2, h // 2
        joints_gt[:, 0] *= w
        joints_gt[:, 1] *= h
        joints_gt += np.array([center_x, center_y])

        return joints_gt

    @staticmethod
    def resize_img(image, joints_gt, size):

        if image.shape[1] == 0 or image.shape[0] == 0:
            print(image.shape)
            return None, None

        fx, fy = size / image.shape[1], size / image.shape[0]

        if fx == 0 or fy == 0:
            print(image.shape)
            print(joints_gt)
            return None, None

        cx, cy = image.shape[1] // 2, image.shape[0] // 2

        joint_vectors = joints_gt - np.array([cx, cy])

        image = cv.resize(image, None, fx=fx, fy=fy)
        joint_vectors *= np.array([fx, fy])
        center_x, center_y = cx * fx, cy * fy
        joints_gt = joint_vectors + np.array([center_x, center_y])

        return image, joints_gt

    @staticmethod
    def flip(image, joints_gt):
        if joints_gt[0, 0] < joints_gt[3, 0]:
            image = cv.flip(image, 1)
            joints_gt[:, 0] = image.shape[1] - joints_gt[:, 0]

        return image, joints_gt

    @staticmethod
    def _make_gaussian(size, sigma=3, center=None):

        x = np.arange(0, size, 1, float)
        y = np.arange(0, size, 1, float)[:, np.newaxis]
        if center is None:
            x0 = size // 2
            y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    @staticmethod
    def generate_hm(size, joints, max_length):

        num_joints = joints.shape[0]
        hm = np.zeros((size, size, num_joints), dtype=np.float32)
        for i in range(num_joints):
            s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
            x = LSPPreProcess._make_gaussian(size=size,
                                             sigma=s,
                                             center=(joints[i, 0], joints[i, 1]))
            hm[:, :, i] = x

        return hm

    @staticmethod
    def get_joint_from_hm(hm):

        joint_coord = np.zeros((FLAGS.no_of_joints, FLAGS.coord_per_joint))
        for i in range(FLAGS.no_of_joints):
            joint_coord[i] = np.unravel_index(hm[:, :, i].argmax(), hm[:, :, i].shape)

        return joint_coord

    @staticmethod
    def resize_joints(joints_gt, output_size, input_size):

        f_x_y = input_size / output_size
        c_x_y = output_size / 2
        center = c_x_y * f_x_y

        joint_vectors = joints_gt - np.array([center, center])
        joint_vectors /= np.array([f_x_y, f_x_y])

        joints_gt = joint_vectors + np.array([c_x_y, c_x_y])

        return joints_gt

    def __next__(self):

        try:

            is_test = self.annotations['examples'][0]['istest'][self.__index]

        except IndexError:
            print('is_train: {}, count: {}'.format(self.is_train, self.__count))
            self.__index = 0
            self.__count = 0
            raise StopIteration

        if self.is_train is True and is_test == 1:
            self.__index += 1
            return None, None
        if self.is_train is False and is_test == 0:
            self.__index += 1
            return None, None

        image = misc.imread(self.images_dir + self.annotations['examples'][0]['filepath'][self.__index][0])
        joints_gt = self.annotations['examples'][0]['coords'][self.__index]
        # joints_gt = joints_gt[:, [0, 1, 2, 3, 4, 5, 0, 3, 6, 9, 12, 13, 16]].transpose()
        joints_gt = joints_gt[:, [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 16]].transpose()

        # check if correct
        # _, axarr = plt.subplots(2, 1)
        # axarr[1].imshow(misc.toimage(image))

        bbox, _ = self.calc_joints_bbox(joints_gt, image)

        image, joints_gt = self.crop_image(image, joints_gt, bbox)

        image, joints_gt = self.resize_img(image, joints_gt,
                                           FLAGS.resize_input_image)

        image, joints_gt = self.flip(image, joints_gt)

        # data aug to be applied on the fly
        # image, joints_gt = self.data_aug(image, joints_gt)

        # # dense posing
        # joints_gt = self.joints_gt_dense(joints_gt)
        #
        # check if correct
        # joints_gt = self.coord_de_normalize(joints_gt)
        # print(joints_gt)
        # axarr[0].imshow(misc.toimage(image))
        # axarr[0].scatter(np.delete(joints_gt[:, 0], [1, 15]),
        #                  np.delete(joints_gt[:, 1], [1, 15]))
        # axarr[0].scatter(joints_gt[:, 0], joints_gt[:, 1])
        # plt.show()

        self.__count += 1
        self.__index += 1

        return np.asarray(image, dtype=np.float32), \
            np.asarray(joints_gt, dtype=np.float32), \


    def __iter__(self):
        return self

