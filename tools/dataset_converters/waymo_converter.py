# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError('Please run "pip install waymo-open-dataset-tf-2-6-0" '
                      '>1.4.5 to install the official devkit first.')

import copy
import os
import os.path as osp
from glob import glob
from io import BytesIO
from os.path import exists, join

import mmengine
import numpy as np
import tensorflow as tf
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection

from mmdet3d.datasets.convert_utils import post_process_coords
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, points_cam2img


class Waymo2KITTI(object):
    """Waymo to KITTI converter.There are 2 steps as follows: Step 1. Extract
    camera images and lidar point clouds from waymo raw data in '*.tfreord' and
    save as kitti format. Step 2. Generate waymo train/val/test infos and save
    as pickle file.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
        test_mode (bool, optional): Whether in the test_mode.
            Defaults to False.
        save_cam_sync_instances (bool, optional): Whether to save cam sync
            instances. Defaults to True.
        save_cam_instances (bool, optional): Whether to save cam instances.
            Defaults to False.
        info_prefix (str, optional): Prefix of info filename.
            Defaults to 'waymo'.
        max_sweeps (int, optional): Max length of sweeps. Defaults to 10.
        split (str, optional): Split of the data. Defaults to 'training'.
    """  # noqa

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False,
                 save_cam_sync_instances=True,
                 save_cam_instances=True,
                 info_prefix='waymo',
                 max_sweeps=10,
                 split='training'):
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            '_FRONT',
            '_FRONT_LEFT',
            '_FRONT_RIGHT',
            '_SIDE_LEFT',
            '_SIDE_RIGHT',
        ]
        self.lidar_list = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]

        # MMDetection3D unified camera keys & class names
        self.camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_SIDE_LEFT',
            'CAM_SIDE_RIGHT',
        ]
        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode
        self.save_cam_sync_instances = save_cam_sync_instances
        self.save_cam_instances = save_cam_instances
        self.info_prefix = info_prefix
        self.max_sweeps = max_sweeps
        self.split = split

        # TODO: Discuss filter_empty_3dboxes and filter_no_label_zone_points
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True
        self.save_track_id = False

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))

        self.image_save_dir = f'{self.save_dir}/image_'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'

        # Create folder for saving KITTI format camera images and
        # lidar point clouds.
        if 'testing_3d_camera_only_detection' not in self.load_dir:
            mmengine.mkdir_or_exist(self.point_cloud_save_dir)
        for i in range(5):
            mmengine.mkdir_or_exist(f'{self.image_save_dir}{str(i)}')

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # mmengine.track_progress(self.convert_one, range(len(self)))
        data_infos = mmengine.track_parallel_progress(self.convert_one,
                                                      range(len(self)),
                                                      self.workers)
        data_list = []
        for data_info in data_infos:
            data_list.extend(data_info)
        metainfo = dict()
        metainfo['dataset'] = 'waymo'
        metainfo['version'] = '1.4'
        metainfo['info_version'] = '1.1'
        waymo_infos = dict(data_infos=data_list, metainfo=metainfo)
        filenames = osp.join(
            osp.dirname(self.save_dir),
            f'{self.info_prefix}_infos_{self.split}.pkl')
        mmengine.dump(waymo_infos, filenames)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert one '*.tfrecord' file to kitti format. Each file stores all
        the frames (about 200 frames) in current scene. We treat each frame as
        a sample, save their images and point clouds in kitti format, and then
        create info for all frames.

        Args:
            file_idx (int): Index of the file to be converted.

        Returns:
            file_infos (list): Waymo infos for all frames in current file.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        # NOTE: file_infos is not shared between processes, only stores frame
        # infos within the current file.
        file_infos = []
        for frame_idx, data in enumerate(dataset):

            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # Step 1.
            self.save_image(frame, file_idx, frame_idx)
            self.save_lidar(frame, file_idx, frame_idx)

            # Step 2.
            # TODO save the depth image for waymo challenge solution.
            self.create_waymo_info_file(frame, file_idx, frame_idx, file_infos)
        return file_infos

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.jpg'
            with open(img_path, 'wb') as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, seg_labels, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        if range_image_top_pose is None:
            # the camera only split doesn't contain lidar points.
            return
        # First return
        points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

        # Second return
        points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

        # timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, mask_indices))

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, frame, file_idx, frame_idx, cam_sync=False):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
            cam_sync (bool, optional): Whether to save the cam sync labels.
                Defaults to False.
        """
        label_all_path = f'{self.label_all_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'
        if cam_sync:
            label_all_path = label_all_path.replace('label_',
                                                    'cam_sync_label_')
        fp_label_all = open(label_all_path, 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for proj_cam in self.cam_list:
                if id + proj_cam in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + proj_cam)
                    name = str(id_to_name.get(id + proj_cam))
                    break

            # NOTE: the 2D labels do not have strict correspondence with
            # the projected 2D lidar labels
            # e.g.: the projected 2D labels can be in camera 2
            # while the most_visible_camera can have id 4
            if cam_sync:
                if obj.most_visible_camera_name:
                    name = str(
                        self.cam_list.index(
                            f'_{obj.most_visible_camera_name}'))
                    box3d = obj.camera_synced_box
                else:
                    continue
            else:
                box3d = obj.box

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = box3d.height
            width = box3d.width
            length = box3d.length

            # TODO: Discuss if we should keep kitti_format
            x = box3d.center_x
            y = box3d.center_y
            z = box3d.center_z - height / 2

            # # project bounding box to the virtual reference frame
            # pt_ref = self.T_velo_to_front_cam @ \
            #     np.array([x, y, z, 1]).reshape((4, 1))
            # x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = box3d.heading
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    truncated, occluded, alpha,
                    bounding_box[0], bounding_box[1],
                    bounding_box[2], bounding_box[3],
                    width, height, length,
                    x, y, z, rotation_y)

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            # Save num_lidar_points_in_box
            line_all = line_all[:-1] + ' ' + str(
                obj.num_lidar_points_in_box) + '\n'

            # label_path = f'{self.label_save_dir}{name}/{self.prefix}' + \
            #     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'
            # if cam_sync:
            #     label_path = label_path.replace('label_', 'cam_sync_label_')
            # fp_label = open(label_path, 'a')
            # fp_label.write(line)
            # fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/{self.prefix}' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def save_timestamp(self, frame, file_idx, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        with open(
                join(f'{self.timestamp_save_dir}/{self.prefix}' +
                     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
                'w') as f:
            f.write(str(frame.timestamp_micros))

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            if c.name == 1:
                mask_index = (ri_index * range_image_mask.shape[0] +
                              mask_index[:, 0]
                              ) * range_image_mask.shape[1] + mask_index[:, 1]
                mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            else:
                mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, intensity, elongation, mask_indices

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

    def create_waymo_info_file(self, frame, file_idx, frame_idx, file_infos):
        frame_infos = dict()

        # Gather frame infos
        sample_idx = \
            f'{self.prefix}{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}'
        frame_infos['sample_idx'] = int(sample_idx)
        frame_infos['timestamp'] = frame.timestamp_micros
        frame_infos['ego2global'] = np.array(frame.pose.transform).reshape(
            4, 4).astype(np.float32).tolist()
        frame_infos['context_name'] = frame.context.name

        # Gather camera infos
        frame_infos['images'] = dict()
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        Tr_velo_to_cams = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            Tr_velo_to_cams.append(Tr_velo_to_cam)

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calibs.append(camera_calib)

        for cam_key, camera_calib, Tr_velo_to_cam, img in zip(
                self.camera_types, camera_calibs, Tr_velo_to_cams,
                frame.images):
            cam_infos = dict()
            cam_infos['img_path'] = str(sample_idx) + '.jpg'
            width, height = Image.open(BytesIO(img.image)).size
            cam_infos['height'] = height
            cam_infos['width'] = width
            cam_infos['lidar2cam'] = Tr_velo_to_cam.astype(np.float32).tolist()
            cam_infos['cam2img'] = camera_calib.astype(np.float32).tolist()
            cam_infos['lidar2img'] = (camera_calib @ Tr_velo_to_cam).astype(
                np.float32).tolist()
            frame_infos['images'][cam_key] = cam_infos

        # Gather lidar infos
        lidar_infos = dict()
        lidar_infos['lidar_path'] = str(sample_idx) + '.bin'
        lidar_infos['num_pts_feats'] = 6
        frame_infos['lidar_points'] = lidar_infos

        # Gather lidar sweeps and camera sweeps infos
        # TODO: Add lidar2img in image sweeps infos when we need it.
        # TODO: Consider merging lidar sweeps infos and image sweeps infos.
        lidar_sweeps_infos, image_sweeps_infos = [], []
        for prev_offset in range(-1, -self.max_sweeps - 1, -1):
            prev_lidar_infos = dict()
            prev_image_infos = dict()
            if frame_idx + prev_offset >= 0:
                prev_frame_infos = file_infos[prev_offset]
                prev_lidar_infos['timestamp'] = prev_frame_infos['timestamp']
                prev_lidar_infos['ego2global'] = prev_frame_infos['ego2global']
                lidar_path = prev_frame_infos['lidar_points']['lidar_path']
                prev_lidar_infos['lidar_path'] = lidar_path
                lidar_sweeps_infos.append(prev_lidar_infos)

                prev_image_infos['timestamp'] = prev_frame_infos['timestamp']
                prev_image_infos['ego2global'] = prev_frame_infos['ego2global']
                prev_image_infos['images'] = dict()
                for cam_key in self.camera_types:
                    prev_image_infos['images'][cam_key] = dict()
                    img_path = prev_frame_infos['images'][cam_key]['img_path']
                    prev_image_infos['images'][cam_key]['img_path'] = img_path
                image_sweeps_infos.append(prev_image_infos)
        if lidar_sweeps_infos:
            frame_infos['lidar_sweeps'] = lidar_sweeps_infos
        if image_sweeps_infos:
            frame_infos['images_sweeps'] = image_sweeps_infos

        if not self.test_mode:
            # Gather instances infos which is used for lidar-based 3D detection
            frame_infos['instances'] = self.gather_instance_info(frame)
            # Gather cam_sync_instances infos which is used for image-based
            # (multi-view) 3D detection.
            if self.save_cam_sync_instances:
                frame_infos['cam_sync_instances'] = self.gather_instance_info(
                    frame, cam_sync=True)
            # Gather cam_instances infos which is used for image-based
            # (monocular) 3D detection (optional).
            if self.save_cam_instances:
                frame_infos['cam_instances'] = self.gather_cam_instance_info(
                    copy.deepcopy(frame_infos['cam_sync_instances']),
                    frame_infos['images'])
        file_infos.append(frame_infos)

    def gather_instance_info(self, frame, cam_sync=False):
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        group_id = 0
        instance_infos = []
        for obj in frame.laser_labels:
            instance_info = dict()
            bounding_box = None
            name = None
            id = obj.id
            for proj_cam in self.cam_list:
                if id + proj_cam in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + proj_cam)
                    name = id_to_name.get(id + proj_cam)
                    break

            # NOTE: the 2D labels do not have strict correspondence with
            # the projected 2D lidar labels
            # e.g.: the projected 2D labels can be in camera 2
            # while the most_visible_camera can have id 4
            if cam_sync:
                if obj.most_visible_camera_name:
                    name = self.cam_list.index(
                        f'_{obj.most_visible_camera_name}')
                    box3d = obj.camera_synced_box
                else:
                    continue
            else:
                box3d = obj.box

            if bounding_box is None or name is None:
                name = 0
                bounding_box = [0.0, 0.0, 0.0, 0.0]

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue
            else:
                label = self.selected_waymo_classes.index(my_type)

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            group_id += 1
            instance_info['group_id'] = group_id
            instance_info['camera_id'] = name
            instance_info['bbox'] = bounding_box
            instance_info['bbox_label'] = label

            height = box3d.height
            width = box3d.width
            length = box3d.length

            # NOTE: We save the bottom center of 3D bboxes.
            x = box3d.center_x
            y = box3d.center_y
            z = box3d.center_z - height / 2

            rotation_y = box3d.heading

            instance_info['bbox3d'] = np.array(
                [x, y, z, length, width, height,
                 rotation_y]).astype(np.float32).tolist()
            instance_info['bbox_label_3d'] = label
            instance_info['num_lidar_pts'] = obj.num_lidar_points_in_box

            if self.save_track_id:
                instance_info['track_id'] = obj.id
            instance_infos.append(instance_info)
        return instance_infos

    def gather_cam_instance_info(self, cam_sync_instances: dict, images: dict):

        cam_instances = dict()
        for cam_type in self.camera_types:
            lidar2cam = np.array(images[cam_type]['lidar2cam'])
            cam2img = np.array(images[cam_type]['cam2img'])
            cam_instances[cam_type] = []
            for instance in cam_sync_instances:
                cam_instance = dict()
                gt_bboxes_3d = np.array(instance['bbox3d'])
                # Convert lidar coordinates to camera coordinates
                gt_bboxes_3d = LiDARInstance3DBoxes(
                    gt_bboxes_3d[None, :]).convert_to(
                        Box3DMode.CAM, lidar2cam, correct_yaw=True)
                corners_3d = gt_bboxes_3d.corners.numpy()
                corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
                in_camera = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d = corners_3d[:, in_camera]
                # Project 3d box to 2d.
                corner_coords = view_points(corners_3d, cam2img,
                                            True).T[:, :2].tolist()

                # Keep only corners that fall within the image.
                final_coords = post_process_coords(
                    corner_coords,
                    imsize=(images[cam_type]['width'],
                            images[cam_type]['height']))

                # Skip if the convex hull of the re-projected corners
                # does not intersect the image canvas.
                if final_coords is None:
                    continue
                else:
                    min_x, min_y, max_x, max_y = final_coords

                cam_instance['bbox'] = [min_x, min_y, max_x, max_y]
                cam_instance['bbox_label'] = instance['bbox_label']
                cam_instance['bbox3d'] = gt_bboxes_3d.numpy().astype(
                    np.float32).tolist()
                cam_instance['bbox_label_3d'] = instance['bbox_label_3d']

                center_3d = gt_bboxes_3d.gravity_center.numpy()
                center_2d_with_depth = points_cam2img(
                    center_3d, lidar2cam, with_depth=True)
                center_2d_with_depth = center_2d_with_depth.squeeze().tolist()

                # normalized center2D + depth
                # if samples with depth < 0 will be removed
                if center_2d_with_depth[2] <= 0:
                    continue
                cam_instance['center_2d'] = center_2d_with_depth[:2]
                cam_instance['depth'] = center_2d_with_depth[2]

                # TODO: Discuss whether following info is necessary
                cam_instance['bbox_3d_isvalid'] = True
                cam_instance['velocity'] = -1
            cam_instances[cam_type].append(cam_instance)

        return cam_instances


def create_ImageSets_img_ids(root_dir, splits):
    save_dir = join(root_dir, 'ImageSets/')
    if not exists(save_dir):
        os.mkdir(save_dir)

    idx_all = [[] for i in splits]
    for i, split in enumerate(splits):
        path = join(root_dir, splits[i], 'calib')
        if not exists(path):
            RawNames = []
        else:
            RawNames = os.listdir(path)

        for name in RawNames:
            if name.endswith('.txt'):
                idx = name.replace('.txt', '\n')
                idx_all[int(idx[0])].append(idx)
        idx_all[i].sort()

    open(save_dir + 'train.txt', 'w').writelines(idx_all[0])
    open(save_dir + 'val.txt', 'w').writelines(idx_all[1])
    open(save_dir + 'trainval.txt', 'w').writelines(idx_all[0] + idx_all[1])
    # open(save_dir + 'test.txt', 'w').writelines(idx_all[2])
    # open(save_dir+'test_cam_only.txt','w').writelines(idx_all[3])
    print('created txt files indicating what to collect in ', splits)
