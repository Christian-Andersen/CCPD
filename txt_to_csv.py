import os
import pandas as pd

for split in ['train', 'test', 'val']:
    df_list = []
    with open(os.path.join('CCPD2019', 'splits', split+'.txt')) as file:
        for line in file:
            cont = False
            line_list = []
            line_list.append(line.split('\n')[0])
            # line_list.append(line.split('-')[0].split('/')[-1])
            bounding_box = line.split('-')[3].split('_')
            x = []
            y = []
            for i in bounding_box:
                x.append(int(i.split('&')[0]))
                y.append(int(i.split('&')[1]))
            A = abs((x[0]*y[1] - x[1]*y[0]) + (x[1]*y[2] - x[2]*y[1]) + (x[2]*y[3] - x[3]*y[2]) + (x[3]*y[0] - x[0]*y[3]))/2
            line_list.append(A)
            for i in range(4):
                if (x[i]<10) or (y[i]<10):
                    cont = True
                line_list.append(x[i])
                line_list.append(y[i])
            if cont:
                continue
            plate = line.split('-')[4].split('_')
            for i in plate:
                line_list.append(i)
            df_list.append(line_list)
    df = pd.DataFrame(df_list, columns=['name',
                                        'area',
                                        'bottom_right_x',
                                        'bottom_right_y',
                                        'bottom_left_x',
                                        'bottom_left_y',
                                        'top_left_x',
                                        'top_left_y',
                                        'top_right_x',
                                        'top_right_y',
                                        'plate_0',
                                        'plate_1',
                                        'plate_2',
                                        'plate_3',
                                        'plate_4',
                                        'plate_5',
                                        'plate_6'])
    df.to_csv(split+'.csv', index=False)
