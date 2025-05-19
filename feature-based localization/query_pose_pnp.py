import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import pyvista as pv
import Metashape as ms
import matplotlib.pyplot as plt
from lightglue.utils import load_image, rbd
from lightglue import LightGlue, SuperPoint, viz2d


def map_to_sphere(x, y, z, pitch, yaw):
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch) +
                            np.cos(theta) * np.cos(pitch))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch) -
                           np.cos(theta) * np.sin(pitch),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def polar_to_cartesian(r, theta, phi):
    return np.array([r * np.cos(phi) * np.cos(theta),
                     r * np.cos(phi) * np.sin(theta),
                     r * np.sin(phi)]).T


def triangulate(pts, ns):
    dirs_mat = ns[:, :, np.newaxis] @ ns[:, np.newaxis, :]
    points_mat = pts[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),
        rcond=None
    )[0]


if __name__ == "__main__":
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'
    GSV_meta_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\GSV_meta'
    query_fd = 'BA_temp\\Monroe_Luma_Rectified_Left_300\\IMG'
    ms_prj_fp = 'spherical_align\\temp.psx'
    query_fp = os.path.join(query_fd, 'template.jpg')
    pose_fp = 'spherical_align\\camera_pose.txt'
    pose_df = pd.read_csv(pose_fp, index_col='#Label')
    matching_rst = pd.read_csv(os.path.join('data\\matching_rst\\Monroe_Luma_Rectified_Left_300',
                                            'global_matching.csv'), index_col='GSV_label')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mkpt = 1024
    extractor = SuperPoint(max_num_keypoints=mkpt).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    img_q = load_image(query_fp)
    feats_q = extractor.extract(img_q.to(device))
    feats_q_uvs = feats_q['keypoints'].cpu().detach().numpy()[0]

    doc = ms.Document()
    doc.open(ms_prj_fp)
    chunk = doc.chunks[0]
    FOV = (74, 50)
    H, W = (1200, 1920)
    f = (W / 2) / np.tan(np.deg2rad(FOV[0] / 2))
    match_lines, match_points, camera_loc = [], [], []

    dense_cloud = chunk.point_cloud
    crs = chunk.crs
    T = chunk.transform.matrix

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    np.random.seed(42)
    tie_real = np.loadtxt('real_tie.txt')
    label, pose, tie_2D = [], [], {}

    for camera_idx, camera in enumerate(chunk.cameras):
        GSV_fp = os.path.join(query_fd, f'{camera.label}.jpg')
        GSV_meta_fp = os.path.join(GSV_meta_fd, f'{camera.label}.metadata.json')

        meta = json.loads(open(GSV_meta_fp, 'r').read())
        northing = np.deg2rad(meta['rotation'])
        pitch, yaw = matching_rst.loc[camera.label, 'u_deg'], matching_rst.loc[camera.label, 'v_deg']

        img_g = load_image(GSV_fp)
        feats_g = extractor.extract(img_g.to(device))
        matches_qg = matcher({"image0": feats_q, "image1": feats_g})
        feats_q_, feats_g_, matches_qg = [
            rbd(x) for x in [feats_q, feats_g, matches_qg]
        ]
        matches = matches_qg["matches"].cpu().detach().numpy()
        kpt_g_uv = feats_g['keypoints'].cpu().detach().numpy()[0, matches[:, 1], :]

        u, v = kpt_g_uv[:, 0], kpt_g_uv[:, 1]
        x, y, z = u - W / 2, H / 2 - v, f
        phi, theta = map_to_sphere(x, y, z, np.deg2rad(90 - yaw), np.deg2rad(pitch - 90))
        Wi, Hi, = camera.sensor.width, camera.sensor.height
        U, V = theta * Wi / (2 * np.pi), phi * Hi / np.pi

        points, feats_2D = [], []
        for idx, qid in enumerate(matches[:, 0]):
            pt_2d = ms.Vector([U[idx], V[idx]])
            v_model = dense_cloud.pickPoint(camera.center, camera.unproject(pt_2d))
            if v_model != None:
                v_world = crs.project(T.mulp(v_model))
                points.append([qid, v_world[0], v_world[1], v_world[2]])
                feats_2D.append([qid, u[idx], v[idx]])
        points = np.array(points)
        match_points.append(np.array(points))

        ox, oy, oz = pose_df.loc[f'{camera.label}.jpg', ['X_est', 'Y_est', 'Z_est']].to_numpy()
        camera_loc.append([ox, oy, oz])
        tie_2D[camera.label] = np.array(feats_2D)
        label.append(f'{camera.label}.jpg')
        pose.append([ox, oy, oz, yaw, 0, meta['rotation'] - 90 - pitch])

        # polar_coord = np.array([(1 / 2) * np.pi - theta + northing, np.pi / 2 - phi]).T
        # vs = polar_to_cartesian(1, polar_coord[:, 0], polar_coord[:, 1])
        #
        # lines = []
        # for idx, qid in enumerate(matches[:, 0]):
        #     item = [qid, ox, oy, oz, vs[idx, 0], vs[idx, 1], vs[idx, 2]]
        #     lines.append(item)
        # match_lines.append(np.array(lines))

    #     raw_GSV_fp = os.path.join(GSV_fd, f'{camera.label}.jpg')
    #     raw_GSV_img = cv2.imread(raw_GSV_fp, cv2.IMREAD_GRAYSCALE)
    #     color = np.random.rand(3,)
    #     for vid, item in enumerate(vs):
    #         ax.plot3D(np.array([ox, ox + item[0]]),
    #                   np.array([oy, oy + item[1]]),
    #                   zs=np.array([oz, oz + item[2]]), color=color)
    #
    #     raw_GSV_img = cv2.imread(raw_GSV_fp, cv2.IMREAD_GRAYSCALE)
    #     U = phi * raw_GSV_img.shape[1] / (2 * np.pi)
    #     V = theta * raw_GSV_img.shape[0] / np.pi
    #     U, V = np.int32(U.flatten()), np.int32(V.flatten())
    #
    #     plt.imshow(raw_GSV_img)
    #     plt.scatter(U, V, s=10, marker='^', c='r')
    #
    #     kptsq, kptsg = feats_q_["keypoints"], feats_g_["keypoints"]
    #     m_kptsq, m_kptsg = kptsq[matches[..., 0]], kptsg[matches[..., 1]]
    #     axes = viz2d.plot_images([img_q, img_g])
    #     viz2d.plot_matches(m_kptsq, m_kptsg, color="lime", lw=0.2)
    #     viz2d.add_text(0, f'Stop after {matches_qg["stop"]} layers')
    # plt.show()

    match_points = [item for item in match_points if len(item) > 0]
    camera_loc = np.array(camera_loc)
    # feats_q_ids = np.concatenate([item[:, 0] for item in match_lines])
    feats_q_ids = np.concatenate([item[:, 0] for item in match_points])
    feats_q_ids, frequency = np.unique(feats_q_ids, return_counts=True)
    feats_q_ids = feats_q_ids[frequency > 2]

    feats_2D, feats_3D, query_tie_2D = [], [], []
    for feats_q_idx in feats_q_ids:
        pts = []

        for points in match_points:
            if feats_q_idx in points[:, 0]:
                pts.append(points[points[:, 0] == feats_q_idx, 1:][0])

        # ns = []
        # for lines in match_lines:
        #     if feats_q_idx in lines[:, 0]:
        #         pts.append(lines[lines[:, 0] == feats_q_idx, 1:4][0])
        #         ns.append(lines[lines[:, 0] == feats_q_idx, 4:][0])
        # pts, ns = np.array(pts), np.array(ns)
        # tie = triangulate(pts, ns).T[0]

        feats_2D.append(feats_q_uvs[int(feats_q_idx)])
        query_tie_2D.append([feats_q_idx, feats_q_uvs[int(feats_q_idx)][0], feats_q_uvs[int(feats_q_idx)][1]])
        pts_mean = np.array(pts).mean(axis=0)
        feats_3D.append([feats_q_idx, pts_mean[0], pts_mean[1], pts_mean[2]])

    query_tie_2D = np.array(query_tie_2D)
    np.savetxt(f'BA_test\\SV_query.csv', query_tie_2D, delimiter=',', fmt='%f')

    for key in tie_2D.keys():
        ids = []
        for idx in tie_2D[key][:, 0]:
            if idx in feats_q_ids:
                ids.append(True)
            else:
                ids.append(False)
        np.savetxt(f'BA_test\\{key}.csv', tie_2D[key][ids], delimiter=',', fmt='%f')

    feats_2D, feats_3D = np.array(feats_2D), np.array(feats_3D)
    pose = np.array(pose)
    pose_out_df = pd.DataFrame({'label': label,
                                'x': pose[:, 0],
                                'y': pose[:, 1],
                                'z': pose[:, 2],
                                'omega': pose[:, 3],
                                'phi': pose[:, 4],
                                'kappa': pose[:, 5]})
    pose_out_df.to_csv(f'BA_test\\pose.csv', index=False)
    np.savetxt(f'BA_test\\tie_3D.csv', feats_3D, delimiter=',', fmt='%f')
    centroid = camera_loc.mean(axis=0)

    obj_3D = feats_3D - centroid
    CM = np.array([[1290.19885253906250000, 0.00000000000000000, 938.12609863281250000],
                   [0.00000000000000000, 1290.06115722656250000, 616.17443847656250000],
                   [0.00000000000000000, 0.00000000000000000, 1.00000000000000000]])
    [ret, rvec, tvec, inliers] = cv2.solvePnPRansac(obj_3D, feats_2D, CM, distCoeffs=None)

    tie_pv = pv.PolyData(tie_real - centroid)
    plotter = pv.Plotter()
    plotter.add_mesh(tie_pv, color='green', point_size=5.0, render_points_as_spheres=True)
    plotter.add_point_labels(pv.PolyData(feats_3D - centroid), np.int32(feats_q_ids).tolist(), point_size=10, font_size=24)
    plotter.show()

    img_q = cv2.imread(query_fp, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_q, cmap='gray')
    plt.scatter(feats_2D[:, 0], feats_2D[:, 1], s=10, marker='^', c='b')
    for idx in range(len(feats_2D)):
        plt.text(x=feats_2D[idx, 0], y=feats_2D[idx, 1], s=str(int(feats_q_ids[idx])), color='r', size='small')

    proj_cv, _ = cv2.projectPoints(obj_3D, rvec, tvec, CM, distCoeffs=None)
    proj_cv = proj_cv.reshape(-1, 2)

    plt.imshow(img_q, cmap='gray')
    plt.scatter(feats_2D[:, 0], feats_2D[:, 1], s=10, marker='^', c='b')
    plt.scatter(proj_cv[:, 0], proj_cv[:, 1], s=10, marker='o', c='r')
    c = 1

