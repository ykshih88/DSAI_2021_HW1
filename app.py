import pandas as pd
import numpy as np
import torch

def data_preprocessing(oper_re,oper_re_percent,time,days=16):
    H_W = int(days**(1/2))
    oper_re_map = np.array(oper_re).reshape((H_W,H_W))[np.newaxis]
    oper_re_percent_map = np.array(oper_re_percent).reshape((H_W,H_W))[np.newaxis]

    data = np.concatenate((oper_re_map,oper_re_percent_map),axis=0)
    time_info = np.array(time%365)
    time_info = np.resize(time_info,(1,4,4))
    data = np.concatenate((data,time_info),axis=0)

    data = torch.tensor(data).float()

    return data
def load_model(model_name='dense_val_best_14851_new'):
    best_model = torch.load(model_name)
    return best_model
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()



    # My code
    df = pd.read_csv(args.training,encoding='utf8')
    oper_re = df['備轉容量(MW)'].to_list()
    oper_re_percent = df['備轉容量率(%)'].to_list()
    time = len(oper_re) + 1

    model = load_model()
    # print(oper_re)
    for i in range(8):
        #因data只到3/21，故預測8天
        data = data_preprocessing(oper_re[-16:],oper_re_percent[-16:],time)

        #predict
        pred_result = model(data.unsqueeze_(0).cuda())

        #result
        oper_re_new = pred_result.squeeze().tolist()
        oper_re += [oper_re_new]
        oper_re_percent += [0]#沒有預測此項補零
        time += 1

    #save csv
    dict_temp  = {}
    dict_temp['date'] = ['20210323','20210324','20210325','20210326','20210327','20210328','20210329']
    dict_temp['operating_reserve(MW)'] = oper_re[-7:]#取最後7天
    dataframe = pd.DataFrame.from_dict(dict_temp)
    dataframe.to_csv(args.output,index=0)
    # print(oper_re)
