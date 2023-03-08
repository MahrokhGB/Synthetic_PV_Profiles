import numpy as np
import matplotlib.pyplot as plt
import sys, os, pvlib, warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with warnings.catch_warnings():
    # NOTE: CSV files for modules contains rows with duplicated names.
    # supppressing duplicated names warnings
    warnings.simplefilter("ignore")
    #Selecting the module database: sandia or CEC
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
    #Selecting the inverter database: CEC or Anton Driesse
    cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')
    adr_inverters = pvlib.pvsystem.retrieve_sam('ADRInverter')



# return list of available modules in each database
def get_available_modules(module_family='sandia'):
    assert module_family.lower() in ['sandia', 'cec']
    if module_family.lower()=='sandia'.lower():
        return list(sandia_modules[sandia_modules.keys()])
    elif module_family.lower()=='cec'.lower():
        return list(cec_modules[cec_modules.keys()])

def get_available_inverters(inverter_family='cec'):
    assert inverter_family.lower() in ['adr', 'cec']
    if inverter_family.lower()=='cec'.lower():
        return list(cec_inverters[cec_inverters.keys()])
    elif inverter_family.lower()=='adr'.lower():
        return list(adr_inverters[adr_inverters.keys()])

def find_module_family(module_name):
    assert not module_name is None
    family = None
    sandia_modules = get_available_modules(module_family='sandia')
    cec_modules = get_available_modules(module_family='cec')
    if module_name in sandia_modules:
        family = 'sandia'
    if module_name in cec_modules:
        if family is None:
            family = 'cec'
        else:
            print('[Error] module found in both families (databases)')
    if family is None:
        print('[Error] module name not found')
    return family


def find_inverter_family(inverter_name):
    assert not inverter_name is None
    family = None
    adr_inverters = get_available_inverters(inverter_family='adr')
    cec_inverters = get_available_inverters(inverter_family='cec')
    if inverter_name in adr_inverters:
        family = 'adr'
    if inverter_name in cec_inverters:
        if family is None:
            family = 'cec'
        else:
            print('[Error] inverter found in both families (databases)')
    if family is None:
        print('[Error] inverter name not found')
    return family


def get_city_info(city_name):
    # returns latitude, longitude, sndaltitude (double) of a city
    if city_name=='Barcelona':
        return {'latitude':41.377, 'longitude':2.188, 'altitude':10}
    elif city_name=='Bern':
        return {'latitude':46.951, 'longitude':7.450, 'altitude':510}
    elif city_name=='Dijon':
        return {'latitude':47.313, 'longitude':5.049, 'altitude':236}
    elif city_name=='Lausanne':
        return {'latitude':46.520, 'longitude':6.632, 'altitude':496}
    elif city_name=='London':
        return {'latitude':51.496, 'longitude':-0.118, 'altitude':15}
    elif city_name=='Milan':
        return {'latitude':45.473, 'longitude':9.185, 'altitude':133}
    elif city_name=='Paris':
        return {'latitude':48.856, 'longitude':2.351, 'altitude':34}
    elif city_name=='Rome':
        return {'latitude':41.894, 'longitude':12.484, 'altitude':36}
    elif city_name=='Tehran':
        return {'latitude':35.689, 'longitude':51.407, 'altitude':1161}
    else:
        print('Info not given for '+ city_name)
        raise NotImplementedError



def visualize_env(env_dict, num_days=5, year=2018, scenario_name=None):
    ''' visualizes data from the given year '''
    # get data tuples and tie series corresponding to the given scenario
    if scenario_name is None:
        scenario_name = list(env_dict['train_scenarios'].keys())[0]
    clients_data = env_dict['train_scenarios'][scenario_name]['clients_data']
    time_series = env_dict['train_scenarios'][scenario_name]['time_series']

    labels = [int(x) for x in env_dict['clients_config']['azimuth']]
    if len(env_dict['city_names'])==1:
        colors = ["#%06x" % np.random.randint(0, 0xFFFFFF) for _ in range(env_dict['num_clients'])]
    else:
        colors =  [(0,1,0), (0,0,1), (1,0,1)]#list(itertools.product((0,1), (0,1), (0,1)))[1:-1]
        #np.random.shuffle(colors)

    # plot num_days days in the given year
    for month in env_dict['months']:
        client_num = 0
        _, axs = plt.subplots(2, 1, figsize = (30, 8*2))
        for mode_num in np.arange(len(env_dict['city_names'])):
            for i in np.arange(env_dict['num_clients_per_mode'][mode_num]):
                # choose color
                if len(env_dict['city_names'])==1:
                    color = colors[client_num]
                else:
                    intensity = 0.2+(1-0.2)*i/(env_dict['num_clients_per_mode'][mode_num]-1) if env_dict['num_clients_per_mode'][mode_num]>1 else 0.5
                    color = tuple(x * intensity for x in colors[mode_num])

                _, y_train, _, y_valid = clients_data[client_num]
                y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0) + 1e-8

                # normalized and unnormalized data
                ts_unnorm = time_series[client_num]
                ts_unnorm = ts_unnorm.loc[(ts_unnorm['month']==month) & (ts_unnorm['year']==year), :]
                ts_unnorm = ts_unnorm.loc[ts_unnorm['hour_day'].isin(env_dict['hours']), :]
                ts_unnorm = ts_unnorm.iloc[np.arange(num_days*len(env_dict['hours'])), :]
                ts_norm = (ts_unnorm.loc[:, 'target'] - y_mean) / y_std

                #if mode_num>-1:
                for day_num in np.arange(num_days):
                    inds_day = day_num*len(env_dict['hours']) + np.arange(len(env_dict['hours']))
                    if day_num==0:
                        axs[0].plot(inds_day, ts_unnorm.loc[:, 'target'].iloc[inds_day],
                                    label=labels[client_num], lw=1, c=color)
                        axs[1].plot(inds_day, ts_norm.iloc[inds_day],
                                    label=labels[client_num], lw=1, c=color)
                    else:
                        axs[0].plot(inds_day, ts_unnorm.loc[:, 'target'].iloc[inds_day],
                                    lw=1, c=color)
                        axs[1].plot(inds_day, ts_norm.iloc[inds_day],
                                    lw=1, c=color)


                client_num += 1
        axs[0].set_ylabel('power unnormalized')
        axs[1].set_ylabel('power normalized')
        for ax in axs:
            ax.set_title(env_dict['info'] + ' - month ' + str(month))
            for i in np.arange(num_days+1):
                ax.axvline(x=len(env_dict['hours'])*i, ymin=0, ymax=1)
            tick_locs = np.linspace(0, num_days*len(env_dict['hours']),
                                    num=int(num_days*len(env_dict['hours'])/4),
                                    endpoint=False).astype(int)
            ax.set_xticks(tick_locs, labels=ts_unnorm.loc[:, 'hour_day'].iloc[tick_locs])
            ax.set_xlabel(str(num_days) + ' days of data')
            ax.legend()
        plt.show()


# ------ PV Panels ------
def visualize_pv(client_ts, pred_mean=None, pred_std=None,
                 selected_months=None, hours=None, num_days=None, figsize=(16, 6)):

    '''
    visulaize predictions for one client
    can be used for plotting predictions either on training or validation data
    - client_ts: time series data of the client in one year
                 (2018 if plotting for training, 2019 for validation)
    - pred_mean: predicted values
    - pred_std: in case of using a GP for making predictions, predicted std. o.w., None
    - selected_months: if training a model only for a subset of months, specify here
                       clients_ts includes all months but pred_mean is only on these months
    - hours: if training a model only for a subset of hours, specify here
             clients_ts includes all hours but pred_mean is only on these months
    - num_days: number of days for visualization. if None, visualizes for the whole year

    '''

    _, ax = plt.subplots(1,1, figsize=figsize)
    if hours is None:
        hours = np.arange(24)
    if selected_months is not None:
        client_ts = client_ts.loc[client_ts['month'].isin(selected_months), :]
    client_ts = client_ts.reset_index()
    if num_days is None:
        ax.plot(list(client_ts.target), c='b', lw=1, label='true')
    else:
        ax.plot(list(client_ts.target)[0:24*num_days], c='b', lw=1, label='true')
    # label with hours
    # plot predictions
    if pred_mean is not None:
        # get indexes of the ts that correspond to predicted steps
        ind = client_ts[client_ts['hour_day'].isin(hours)].index.tolist()

        if num_days is not None: # plot only a fraction of the data
            ind = ind[0:num_days*len(hours)]
            pred_mean=pred_mean[0:num_days*len(hours)]

        ax.scatter(ind, pred_mean, color='r', s=5, label='prediction')
        if pred_std is not None:
            lb = list(client_ts.target)
            ub = list(client_ts.target)
            for i in np.arange(len(ind)):
                lb[ind[i]] = pred_mean[i] - 1.645 * pred_std[i]
                ub[ind[i]] = pred_mean[i] + 1.645 * pred_std[i]
            ax.fill_between(np.arange(len(lb)), lb, ub, label = 'confidence', color='red',
            alpha=0.5)
    plt.show()



def visualize_pv_train_valid(env_dict, scenario_name, client_num, model, figsize=(30, 8*2)):
    time_series = env_dict['train_scenarios'][scenario_name]['time_series']
    for month in env_dict['months']:
        _, axs = plt.subplots(2,1, figsize=figsize)
        # plot train data

        # plot validation data




def normalize_data_tup(data):
    # for one client only
    data_nrm=(None, None, None, None)
    # statistics on train data
    x_mean, y_mean = np.mean(data[0], axis=0), np.mean(data[1], axis=0)
    x_std, y_std = np.std(data[0], axis=0) + 1e-8, np.std(data[1], axis=0) + 1e-8
    # normalize
    data_nrm = ((data[0] - x_mean[None, :]) / x_std[None, :],
                (data[1] - y_mean[None, :]) / y_std[None, :],
                (data[2] - x_mean[None, :]) / x_std[None, :],
                (data[3] - y_mean[None, :]) / y_std[None, :])
    # check sizes
    assert data_nrm[0].shape == data[0].shape
    assert data_nrm[1].shape == data[1].shape
    assert data_nrm[2].shape == data[2].shape
    assert data_nrm[3].shape == data[3].shape

    return data_nrm, y_mean, y_std





def tile_in_list(a, l):
    '''
    a: values
    l: list of length of the output list
    if a is a list and l is a list, repeats a_i for l_i times.
    if a is a single value, repeats a for sum(l) times
    '''
    # convert all to list
    if not isinstance(a,list):
        a=[a]
    if not isinstance(l,list):
        l=[l]
    # total length of the result
    len_tot = sum(l)

    # if a is already the correct length, return it
    if len(a)==len_tot:
        return a
    # if a is a single element, repeat it
    if len(a)==1:
        return [a[0]]*len_tot
    # repeat a_i for l_i times
    if len(a)==len(l):
        res=[]
        for ai, li in zip(a, l):
            res += [ai]*li
        return res
    else:
        print('[ERROR]: should pass a single value, list of values to be tiled, or a full list')
        return []




def tile_in_list(a, l):
    '''
    a: values
    l: list of length of the output list
    if a is a list and l is a list, repeats a_i for l_i times.
    if a is a single value, repeats a for sum(l) times
    '''
    # convert all to list
    if not isinstance(a,list):
        a=[a]
    if not isinstance(l,list):
        l=[l]
    # total length of the result
    len_tot = sum(l)

    # if a is already the correct length, return it
    if len(a)==len_tot:
        return a
    # if a is a single element, repeat it
    if len(a)==1:
        return [a[0]]*len_tot
    # repeat a_i for l_i times
    if len(a)==len(l):
        res=[]
        for ai, li in zip(a, l):
            res += [ai]*li
        return res
    else:
        print('[ERROR]: should pass a single value, list of values to be tiled, or a full list')
        return []



def normalize_data_tup(data):
    # for one client only
    data_nrm=(None, None, None, None)
    # statistics on train data
    x_mean, y_mean = np.mean(data[0], axis=0), np.mean(data[1], axis=0)
    x_std, y_std = np.std(data[0], axis=0) + 1e-8, np.std(data[1], axis=0) + 1e-8
    # normalize
    data_nrm = ((data[0] - x_mean[None, :]) / x_std[None, :],
                (data[1] - y_mean[None, :]) / y_std[None, :],
                (data[2] - x_mean[None, :]) / x_std[None, :],
                (data[3] - y_mean[None, :]) / y_std[None, :])
    # check sizes
    assert data_nrm[0].shape == data[0].shape
    assert data_nrm[1].shape == data[1].shape
    assert data_nrm[2].shape == data[2].shape
    assert data_nrm[3].shape == data[3].shape

    return data_nrm, y_mean, y_std




def visualize_ts(client_ts, pred_mean=None, pred_std=None, title=None,
                 selected_months=None, hours = None, figsize=(16, 6)):

    '''
    visulaize predictions for one client

    USAGE EXAMPLE
    for client_num in np.arange(num_clients):
        title = 'Predictions on training data for client {:2.0f}'.format(client_num)
        _, predictions = clients_train_data[client_num] # assume an ideal model with zero error, in practice, this must be the output of your model on clients_train_data[client_num][0]
        visualize_ts(clients_ts=clients_train_ts[client_num], pred_mean=predictions, selected_months=months, hours=hours, title = title)
    '''

    # check predictions provided for all clients
    #if (pred_mean is not None) or (pred_std is not None):
    #    assert (pred_mean is not None) and (pred_std is not None)
    #    assert len(pred_mean) == len(pred_std)

    fig, ax = plt.subplots(1,1, figsize=figsize)

    if selected_months is not None:
        client_ts = client_ts.loc[client_ts['month'].isin(selected_months), :]
    client_ts = client_ts.reset_index()
    ax.plot(list(client_ts.p_mp), c='b', lw=1, label='true')
    # TODO label with hours
    ax.set_xlabel('time (h)')
    ax.set_xlabel('generated power (kWh/h)')
    if not title is None:
        ax.set_title(title)
    # plot predictions
    if pred_mean is not None and hours is not None:
        # get indexes of the ts that correspond to predicted steps
        ind = client_ts[client_ts['hour_day'].isin(hours)].index.tolist()
        ind = ind[0:pred_mean.shape[0]] # TODO: this is a bug in test_ts. ind and pred means must be the same size
        #ind = [i-1 for i in ind]
        ax.scatter(ind, pred_mean, color='r', s=5, label='prediction')
        if pred_std is not None:
            lb = list(client_ts.p_mp)
            ub = list(client_ts.p_mp)
            for i in np.arange(len(ind)):
                lb[ind[i]] = pred_mean[i] - 1.645 * pred_std[i]
                ub[ind[i]] = pred_mean[i] + 1.645 * pred_std[i]
            ax.fill_between(np.arange(len(lb)), lb, ub, label = 'confidence', color='red',
             alpha=0.5)




from sklearn.metrics import r2_score
def adj_r2_scorer(estimator, X, y):
    y = y.flatten()
    y_pred = estimator.predict(X).flatten()
    R2 = r2_score(y, y_pred)
    n, p = X.shape
    return 1-(1-R2)*(n-1)/(n-p-1)