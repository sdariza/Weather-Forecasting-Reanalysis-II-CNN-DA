import pickle

N_LATS = 73
N_LONS = 144

def get_pred(r, ip, jp):
    x = [] # x_i
    val = ip * N_LONS + jp # predecesorNum
    # print(f'predecesores de {val}')
    j_ = [l % N_LONS for l in range(jp-r,jp+r+1)] # lons dentro del radio
    if ip == 0:
        i_ = [k % N_LATS for k in range(ip,ip+r+1)] # lats dentro del radio lat=90
    elif ip == N_LATS - 1:
        i_ = [k for k in range(ip-r, N_LATS)] # lats dentro del rario lat=-90
    else:
        i_ = [k for k in range(max(0,ip-r), min(N_LATS,ip+r+1))] #lats dentro del radio -90<lats<90
    for ii in i_:
        for jj in j_:
            if ii*N_LONS+jj < val:
                x.append(ii* N_LONS + jj) # number of variables that are predecessors
    # print(f'pred: {x}')
    return x

for r in [3,5,10,15,30,45]:
    predecessors = {}
    cont_p = 0
    for i in range(0, N_LATS):
        for j in range(0, N_LONS):
            predecessors[f'{i*N_LONS+j}'] = get_pred(r,i,j)
            cont_p = cont_p + len(predecessors[f'{i*N_LONS+j}'])
    predecessors['total'] = cont_p

    with open(f"./data_assimilation/predecessors/predecessor_r{r}", "wb") as fp:
        pickle.dump(predecessors, fp)