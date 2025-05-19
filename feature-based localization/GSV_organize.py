import os
import json
import shutil
import pathlib
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    in_fd = 'data\\GSV\\raw'
    out_fp = 'data\\shps\\GSV.csv'
    fps_all = list(pathlib.Path(in_fd).rglob("*.jpg"))
    PIDs, lats, lngs, elevations, northings, years, months, days, label = [], [], [], [], [], [], [], [], []
    if os.path.exists(out_fp):
        pre = True
        GSV_info = pd.read_csv(out_fp)
    else:
        pre = False

    meta_fd = 'data\\GSV\\GSV_meta'
    depth_fd = 'data\\GSV\\depthmap'
    img_fd = 'data\\GSV\\IMG'

    if os.path.exists(meta_fd):
        shutil.rmtree(meta_fd)
    os.mkdir(meta_fd)

    if os.path.exists(depth_fd):
        shutil.rmtree(depth_fd)
    os.mkdir(depth_fd)

    if os.path.exists(img_fd):
        shutil.rmtree(img_fd)
    os.mkdir(img_fd)

    new_idx = GSV_info.shape[0]

    for idx in tqdm(range(len(fps_all)), desc='Organizing GSV patch: '):
        fp = fps_all[idx].__fspath__()

        if 'depth' not in fp:
            meta_fp = fp.replace('.jpg', '.metadata.json')
            depth_fp = fp.replace('.jpg', '.depthmap.jpg')
            meta = json.loads(open(meta_fp, 'r').read())

            year, month = meta['date']['year'], meta['date']['month']
            PID, deg = meta['panoId'], meta['rotation']
            dst_sv_fp = os.path.join(img_fd, f'SV_{new_idx}.jpg')
            dst_meta_fp = os.path.join(meta_fd, f'SV_{new_idx}.metadata.json')
            dst_depth_fp = os.path.join(depth_fd, f'SV_{new_idx}.depthmap.jpg')

            if meta['elevation'] != None and PID not in GSV_info['PID']:
                shutil.copy(fp, dst_sv_fp)
                shutil.copy(meta_fp, dst_meta_fp)
                shutil.copy(depth_fp, dst_depth_fp)
                elevations.append(meta['elevation'])

                try:
                    days.append(meta['date']['day'])
                except:
                    days.append(0)

                PIDs.append(PID)
                lats.append(meta['lat'])
                lngs.append(meta['lng'])
                years.append(year)
                months.append(month)
                northings.append(deg)
                label.append(f'SV_{new_idx}')
                new_idx += 1

    df = pd.DataFrame({'label': label,
                       'PID': PIDs,
                       'latitude': lats,
                       'longitude': lngs,
                       'elevation': elevations,
                       'northing': northings,
                       'year': years,
                       'month': months,
                       'day': days})
    df = pd.concat([GSV_info, df], ignore_index=True)
    df.to_csv('GSV.csv', index=False)
