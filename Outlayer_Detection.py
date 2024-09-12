def Out_detector(x,y):

    import numpy as np
    from sklearn.svm import OneClassSVM
    algorithm=OneClassSVM(kernel='rbf',gamma='auto')
    y_pred=algorithm.fit_predict(x)


    x0=[]
    y0=[]
    x1=[]
    y1=[]
    for i in range(len(y_pred)):
        if y_pred[i]==-1:
            x0.append(x[i])
            y0.append(y[i])
        else:
            x1.append(x[i])
            y1.append(y[i])

    if len(x0)>len(x1):
        ClearData=x0
        ClearLabels=y0
    else:
        ClearData=x1
        ClearLabels=y1

    ClearData=np.array(ClearData)
    ClearLabels=np.array(ClearLabels)
    return ClearData,ClearLabels
