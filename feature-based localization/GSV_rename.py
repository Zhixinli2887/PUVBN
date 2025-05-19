import os
import json
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    out_fp = 'data\\shps\\GSV.csv'
    PIDs, lats, lngs, elevations, northings, years, months, days, label = [], [], [], [], [], [], [], [], []

    meta_fd = 'data\\GSV\\GSV_meta'
    depth_fd = 'data\\GSV\\depthmap'
    img_fd = 'data\\GSV\\IMG'
    fnames = os.listdir(img_fd)

    for idx in tqdm(range(len(fnames)), desc=' Projecting GSV patch: '):
        fname = fnames[idx]
        sv_id = f'SV_{idx}'

        img_fp = os.path.join(img_fd, fname)
        meta_fp = os.path.join(meta_fd, fname.replace('.jpg', '.metadata.json'))
        depth_fp = os.path.join(depth_fd, fname.replace('.jpg', '.depthmap.jpg'))
        meta = json.loads(open(meta_fp, 'r').read())

        year, month = meta['date']['year'], meta['date']['month']
        PID, deg = meta['panoId'], meta['rotation']
        img_fp_new = os.path.join(img_fd, f'SV_{idx}.jpg')
        meta_fp_new = os.path.join(meta_fd, f'SV_{idx}.metadata.json')
        depth_fp_new = os.path.join(depth_fd, f'SV_{idx}.depthmap.jpg')

        os.rename(img_fp, img_fp_new)
        os.rename(meta_fp, meta_fp_new)
        os.rename(depth_fp, depth_fp_new)

        if meta['elevation'] != None:
            label.append(sv_id)
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

    df = pd.DataFrame({'label': label,
                       'PID': PIDs,
                       'latitude': lats,
                       'longitude': lngs,
                       'elevation': elevations,
                       'northing': northings,
                       'year': years,
                       'month': months,
                       'day': days})
    df.to_csv(out_fp, index=False)
