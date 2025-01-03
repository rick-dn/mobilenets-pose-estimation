import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
from flic_pre_processing import LSPPreProcess

from flags import FLAGS


def get_dist_pck(inference, joints_gt):

    ref_dist = np.linalg.norm([joints_gt[6, :] - joints_gt[3, :]])

    dist = np.zeros(FLAGS.no_of_joints)
    for i, _ in enumerate(inference):
        dist[i] = np.linalg.norm([inference[i, :] - joints_gt[i, :]])/ref_dist

    return dist


def evaluate_pck(joints_gt_batch, inference_batch, batch_size):

    # no_of_joints = [batch_size,
    #                 FLAGS.no_of_joints + FLAGS.no_of_dense_joints,
    #                 FLAGS.coord_per_joint]

    no_of_joints = [batch_size, FLAGS.heatmap_size, FLAGS.heatmap_size,
                    FLAGS.no_of_joints + FLAGS.no_of_dense_joints]

    inference_batch = np.array(inference_batch).reshape(no_of_joints)
    joints_gt_batch = np.array(joints_gt_batch).reshape(no_of_joints)

    pck = np.zeros((batch_size, FLAGS.no_of_joints), dtype=int)

    for idx in enumerate(zip(inference_batch, joints_gt_batch)):

        inference = idx[1][0]
        joints_gt = idx[1][1]

        # inference = inference[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12], :]
        # joints_gt = joints_gt[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12], :]

        # inference = LSPPreProcess.coord_de_normalize(inference)
        # joints_gt = LSPPreProcess.coord_de_normalize(joints_gt)

        inference = LSPPreProcess.get_joint_from_hm(inference)
        joints_gt = LSPPreProcess.get_joint_from_hm(joints_gt)

        dist = get_dist_pck(inference=inference, joints_gt=joints_gt)

        for j in range(FLAGS.no_of_joints):
            pck[idx[0], j] = dist[j] < FLAGS.low_thresh

    print('joint wise pck: {}'.format(np.mean(pck, axis=0)))
    print('total pck: {:.5f}'.format(np.mean(np.mean(pck, axis=0))))
    return np.mean(np.mean(pck, axis=0))


def draw_coordinates(org_image_batch, joints_gt_batch, inference_batch, batch_size):
    """This methods is used for displaying inferences onto the images.
    It reshapes the inferences and displays onto original images.
    It also projects the ground truth coordinates for comparison.
    Args:
      org_image_batch: Images
      joints_gt_batch: gt coordinates
      inference_batch: inference coordinates
    Returns:
      image: Flipped  image
      joints_gt: Flipped joints
    """

    no_of_joints = [batch_size, FLAGS.no_of_joints, FLAGS.coord_per_joint]
    inference_batch = np.asarray(inference_batch).reshape(no_of_joints)
    joints_gt_batch = np.asarray(joints_gt_batch).reshape(no_of_joints)

    for i in range(batch_size):

        image = np.asarray(org_image_batch[i])
        inference = inference_batch[i].reshape(FLAGS.no_of_joints, FLAGS.coord_per_joint)
        joints_gt = joints_gt_batch[i].reshape(FLAGS.no_of_joints, FLAGS.coord_per_joint)

        # inference = DataIterator.coord_de_normalize(inference)
        # joints_gt = DataIterator.coord_de_normalize(joints_gt)

        # print('joints_gt:\n', joints_gt)
        # print('inference', inference)

        plt.axis('off')
        plt.imshow(misc.toimage(image))
        plt.scatter(joints_gt[:, 0], joints_gt[:, 1], s=100, c='r')
        plt.scatter(inference[:, 0], inference[:, 1], s=100, c='b')
        plt.show()


def main():
    batch_size = FLAGS.test_batch_size
    # norm_img_batch, joints_gt_batch, inference_batch,  = predict.predict()
    # evaluate_pck(joints_gt_batch=joints_gt_batch,
    #              inference_batch=inference_batch,
    #              batch_size=batch_size)
    # draw_coordinates(norm_img_batch, joints_gt_batch, inference_batch, batch_size)


if __name__ == '__main__':
    main()

