
import numpy as np  

test = np.load('./08_diff.npy')
#print(test[500:600,[3, 7, 11]])
#np.savetxt('./07_diff.txt', test)
 
for line in range(test.shape[0]):

    # cur_Tr = Tr[n0, :, :]

    gt =test[line].reshape(3,4)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    TT = np.concatenate([gt, filler], axis=0)

    if line == 0:
        T_final = TT
        T = T_final[:3, :]
        T = T.reshape( 1, 12)
    else:
        T_final = np.matmul(T_final, TT)   #右乘，左乘  ??
        T_current = T_final[:3, :]
        T_current = T_current.reshape( 1, 12)
        T = np.append(T, T_current, axis=0)

T = T.reshape(-1, 12)
print(T)
