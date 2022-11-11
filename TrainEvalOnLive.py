import numpy as np
import skimage.io
from brisque_feature import brisque_feature
import scipy.io
from tqdm import trange
from sklearn.model_selection import GridSearchCV,cross_validate,ShuffleSplit
from sklearn.svm import SVR
from scipy import stats
#读取LIVE数据库用，非核心函数
def adjust_name(idx,ntype):
    if ntype=='jp2k':
        if idx>227:
            raise ValueError("Image Index Out of Range")
        return idx-1
    elif ntype=='jpeg':
        if idx>233:
            raise ValueError("Image Index Out of Range")
        return idx+226
    elif ntype=='wn':
        if idx>174:
            raise ValueError("Image Index Out of Range")
        return idx+459
    elif ntype=='gblur':
        if idx>174:
            raise ValueError("Image Index Out of Range")
        return idx+633
    elif ntype=='fastfading':
        if idx>174:
            raise ValueError("Image Index Out of Range")
        return idx+807
    else:
        raise ValueError("Unkown Distortion Type")

def CreatSet():
    mat=scipy.io.loadmat('LIVE_dmos2.mat')
    dmoses=mat['dmos_new']
    distortions=['fastfading','gblur','wn','jp2k','jpeg']
    setNum={'fastfading':174,#不同失真类型个数
            'gblur':174,
            'wn':174,
            'jp2k':227,
            'jpeg':233}
    X=[]
    Y=[]
    for distor in distortions:
        path='D:\\DataBase\\IMAGE\\databaserelease2\\'+distor+'\\img'
        for img_id in trange(1,(setNum[distor]+1)):
            img=skimage.io.imread(path+str(img_id)+'.bmp',as_gray=True)
            feat=brisque_feature(img)
            X.append(feat)
            Y.append(dmoses[0,adjust_name(img_id,distor)])
    np.save('feat_Live',X)
    np.save('score_Live',Y)

def evaluate():
    x=np.load('feat_Live.npy')
    y=np.load('score_Live.npy')
    ss = ShuffleSplit(n_splits=1000, random_state=0, test_size=0.2)
    cout_num=0
    SROCC_box=np.zeros((1000,1),dtype=np.float32)
    PLCC_box=np.zeros((1000,1),dtype=np.float32)

    params = [
        {'kernel': ['rbf'], 'C': 2**(np.arange(-8,8,0.8)), 'gamma': 2**(np.arange(-8,8,0.8))}
    ]
    svr = SVR()
    clf = GridSearchCV(svr, params)
    clf.fit(x, y)
    for train_index,test_index in ss.split(x):
        svr = SVR(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
        svr.fit(x[train_index],y[train_index])
        predict=svr.predict(x[test_index])

        SROCC,_=stats.spearmanr(predict, y[test_index])
        PLCC,_ = stats.pearsonr(predict, y[test_index])
        SROCC_box[cout_num,:]=SROCC
        PLCC_box[cout_num,:]=PLCC
        cout_num=cout_num+1
    print('Median SRCC %.4f  MedianPLCC %.4f'%(np.median(SROCC_box),np.median(PLCC_box)))


if __name__=='__main__':
    CreatSet()
    evaluate()

