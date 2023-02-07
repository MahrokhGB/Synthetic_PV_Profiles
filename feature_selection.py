import numpy as np
import statsmodels.api as sm

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

import sys, os, copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------- SMWrapper -----------------------
class SMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept


    def fit(self, X, y,
            feat_names=[]):
        y = y.flatten()
        # initialize
        if self.fit_intercept:
            X = sm.add_constant(X,has_constant='add')
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.model_ = self.model_class(y, X)
        # fit
        self.fitted_model_ = self.model_.fit()
        self.feature_importances_ = self.fitted_model_.tvalues
        # optional
        if not feat_names==[]:
            self.feat_names_=feat_names
        return self


    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')
        # Input validation
        X = check_array(X)
        # predict
        if self.fit_intercept:
            X = sm.add_constant(X,has_constant='add')
        return self.fitted_model_.predict(X).flatten()

    def predict_bis(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')
        # Input validation
        X = check_array(X)
        # predict
        #if self.fit_intercept:
            #X = sm.add_constant(X)
        return self.fitted_model_.predict(X).flatten()



    def get_params(self, deep = False):
        return {'fit_intercept':self.fit_intercept,
               'model_class':self.model_class}


    def summary(self, xname=[]):
        return self.fitted_model_.summary(xname=xname)


    def cv_fit(self, X, y, n_splits=5,
               scoring='r2',
               verbose=True):
        y = y.flatten()
        r2_train = [0]*n_splits
        r2_test = [0]*n_splits
        models = []
        test_inds = []
        # folds
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_num=0
        for train_index, test_index in kf.split(X):
            test_inds.append(test_index)

            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            # fit to this fold
            self.fit(X_train, y_train)
            models.append(self.fitted_model_)
            r2_train[fold_num] = self.fitted_model_.rsquared
            # test scores
            y_test_pred = self.predict(X_test)
            r2_test[fold_num] = r2_score(y_test, y_test_pred)
            fold_num = fold_num+1
        # find best
        if verbose:
            print('Train R2 scores: ', *r2_train)
            print('Test  R2 scores: ', *r2_test)
        maxr2, ind = max((val, idx) for (idx, val) in enumerate(r2_test))
        self.fitted_model_ = models[ind]
        self.test_inds_ = test_inds[ind]
        self.train_inds_ = np.setdiff1d(np.arange(0, len(X[:,0])),
                                        self.test_inds_)
        return self

    def get_weights(self):
        return self.fitted_model_.params

    def get_names(self):
        return self.fitted_model_.feature_names_in_





# ------------------------- ERROR MEASURES ----------------------------------------
# MEASURES
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
def error_measures(model, X, y, verbose=False):
    y = y.flatten()
    # split train test
    X_test = X[model.test_inds_, :]
    y_test = y[model.test_inds_]
    X_train = X[model.train_inds_, :]
    y_train = y[model.train_inds_]

    n = X_train.shape[0]
    p = X_train.shape[1]
    p2 = X_test.shape[1]
    if not p==p2:
        print('error')

    # Make predictions using the testing set
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    assert y_pred_train.shape == y_train.shape
    assert y_pred.shape == y_test.shape
    # MAE
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test  = mean_absolute_error(y_test, y_pred)
    # MSE
    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test  = mean_squared_error(y_test, y_pred)
    # explained var
    EVA_train = explained_variance_score(y_train, y_pred_train)
    EVA_test  = explained_variance_score(y_test, y_pred)
    # The coefficient of determination: 1 is perfect prediction
    R2_train = r2_score(y_train, y_pred_train)
    R2_test = r2_score(y_test, y_pred)
    # adjusted r2
    Adjr2_train = 1-(1-R2_train)*(n-1)/(n-p-1)
    Adjr2_test  = 1-(1-R2_test)*(n-1)/(n-p-1)
    # aic
    AIC_train = -2*math.log(len(y_train)*MSE_train)+2*p
    AIC_test  = -2*math.log(len(y_test)*MSE_test) +2*p

    # save to dict
    meas = {'MAE_train':MAE_train, 'MAE_test':MAE_test,
           'MSE_train':MSE_train, 'MSE_test':MSE_test,
           'EVA_train':EVA_train, 'EVA_test':EVA_test,
           'R2_train':R2_train, 'R2_test':R2_test,
           'Adjr2_train':Adjr2_train, 'Adjr2_test':Adjr2_test,
           'AIC_train':AIC_train, 'AIC_test':AIC_test}

    # print
    if verbose:
        print('Mean absolute error: train %.4f, test %.4f' % (MAE_train, MAE_test))
        print('Mean squared error:  train %.4f, test %.4f' % (MSE_train, MSE_test))
        print('Explained Variance Score (best=1): train %.4f, test %.4f' % (EVA_train, EVA_test))
        print('Coefficient of determination (R2): train %.4f, test %.4f' %(R2_train, R2_test))
        print('Adjusted coeff. of determination:  train %.4f, test %.4f' %(Adjr2_train, Adjr2_test))
        print('AIC: train %.2f, test %.2f' %(AIC_train, AIC_test))
    return meas



# ------------------ PACF ---------------------
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sttools
from house_pv import _augment_lags
def tune_pacf(
    house, max_num_lags, max_num_days,
    train_years, valid_years,
    step_ahead=1, repeats=1, verbose=True):

    '''
    INPUTS: data_power dataframe, maximum number of lags to be considered,
    maximum number of days in the past that these lags should be looked for,
    the number of repeats for each number of feats.

    OUTPUTS:
    '''
    import copy
    # compute pacf
    # NOTE: ideally, should have computed PACF on training data. but filtering to hours would
    # cause lags to have a different meaning than shifts in PACF. e.g., if 7AM to 19PM is selected,
    # lag one at 7 AM is 6 AM but for PACF, the previous value is 19 PM the day before.
    # only filtering to months is done, not to hours. data from both train and validation years
    data_power_org = copy.deepcopy(house.data_power)
    powers = house.data_power.loc[:, 'target'].values
    #powers = house.data_power.loc[house.data_power['month'].isin(house.months), 'target'].values
    pacf_val = sttools.pacf(powers, nlags=max_num_days*24)
    pacf_val = np.abs(pacf_val)             # compute abs
    sorted_lags = np.argsort(-pacf_val)[1:] # remove first lag which is 0

    # build regression df with all lags
    house.data_power = _augment_lags(
                            house.data_power, lags=sorted_lags,
                            step_ahead=step_ahead)
    # add lags to feature names
    for lag in sorted_lags:
        house.feature_names.append('lag ' + str(lag))

    (X_train, y_train, _, _) = house.construct_regression_matrices(
        m_train=None, train_years=train_years, valid_years=valid_years)
    house.data_power = data_power_org

    # remove lags that are constant in training points
    const_lag_inds = []
    const_feat_inds = []
    const_feat_names = []
    num_feats_non_ar = len([x for x in house.feature_names if not x.startswith('lags')])
    for col_num in np.arange(X_train.shape[1]):
        col_std = np.std(X_train[:, col_num])
        col_mean = np.mean(X_train[:, col_num])
        if col_std<=1e-4*col_mean+1e-6:
            const_feat_inds.append(col_num)
            const_feat_names.append(house.feature_names[col_num])
            if house.feature_names[col_num].startswith('lag'):
                const_lag_inds.append(col_num-num_feats_non_ar)

    # remove constant cols from data
    if len(const_feat_names)>0:
        # remove from list of features
        if verbose:
            print('[INFO] the following constnat features were removed: ', *const_feat_names)
        for const_feat in const_feat_names:
            house.feature_names.remove(const_feat)
        # remove from sorted lags
        sorted_lags = np.delete(sorted_lags, const_lag_inds)

        # remove from data
        X_train = np.delete(X_train, const_feat_inds, axis=1)
        assert len(house.feature_names) == X_train.shape[1]

    # init
    r2_adj_train = [[0]*(max_num_lags+1)]*repeats
    r2_adj_test  = [[0]*(max_num_lags+1)]*repeats
    mse_train = [[0]*(max_num_lags+1)]*repeats
    mse_test  = [[0]*(max_num_lags+1)]*repeats
    aic_test  = [[0]*(max_num_lags+1)]*repeats


    selected_feats = []
    selected_feats_inds = []
    # add all non-lag features
    for feature_name in house.feature_names:
        if not feature_name.startswith('lag'):
            selected_feats.append(feature_name)
            selected_feats_inds.append(house.feature_names.index(feature_name))

    # add lags one by one
    for num_lag in np.arange(max_num_lags+1):
        if num_lag>0:
            new_feat_name = 'lag ' + str(sorted_lags[num_lag-1])
            selected_feats.append(new_feat_name)
            selected_feats_inds.append(house.feature_names.index(new_feat_name))
        #print('\nNumber of lags: {:2.0f}'.format(num_lag))
        #print('feature names: ', *selected_feats)

        # fit
        for rep in np.arange(repeats):
            print(X_train[:, selected_feats_inds].shape)
            ols_obj = SMWrapper(sm.OLS, fit_intercept=True).cv_fit(
                            X_train[:, selected_feats_inds],
                            y_train.flatten(), verbose=True
                            )

            # scores
            meas = error_measures(
                            ols_obj,
                            X=X_train[:, selected_feats_inds],
                            y=y_train.flatten(), verbose=True)
            r2_adj_train[rep][num_lag-1] = meas['Adjr2_train']
            r2_adj_test[rep][num_lag-1]  = meas['Adjr2_test']
            mse_train[rep][num_lag-1] = meas['MSE_train']
            mse_test[rep][num_lag-1]  = meas['MSE_test']
            aic_test[rep][num_lag-1]  = meas['AIC_test']
            del ols_obj
    # find lowest and highest
    maxr2_adj, best_num_r = max((val, idx) for (idx, val) in enumerate(np.mean(r2_adj_test,axis=0)))
    best_num_r=best_num_r+1
    min_mse, best_num_m = min((val, idx) for (idx, val) in enumerate(np.mean(mse_test,axis=0)))
    best_num_m=best_num_m+1
    # find best
    best = np.argmax((r2_adj_test>=0.95*maxr2_adj) * (mse_test<=1.1*min_mse))
    best=best+1
    first_in_range=best
    # plot
    plt.style.use('fivethirtyeight')
    x = np.arange(max_num_lags+1)
    # adj R2, aic
    fig,ax = plt.subplots(figsize=(20,10))
    ax.plot(x, np.mean(r2_adj_test,axis=0),
            c='blue',label = 'test adjusted R2', linewidth=3, ms=1)
    ax.fill_between(x, np.mean(r2_adj_test,axis=0)-1.96*np.std(r2_adj_test,axis=0),
                    np.mean(r2_adj_test,axis=0)+1.96*np.std(r2_adj_test,axis=0),
                    color='purple', alpha=0.1, label='conf. bound test adj. R2')
    ax.plot(x, np.mean(r2_adj_train,axis=0),
            'b--',label = 'train adj. R2', linewidth=3, ms=1)
    ax.scatter(best_num_r,maxr2_adj,c='red',label = 'highest test adj. R2', s=100, marker='X')
    ax.set_xlabel('Number of auto-regressors')
    ax.set_ylabel('Adjusted R2 Score')
    ax.set_title('Tuning Number of Auto-Regressors')
    # plot aic
    ax2 = ax.twinx()
    ax2.plot(x, np.mean(mse_test,axis=0),
             c='orange',label = 'MSE test', linewidth=3, ms=1)
    ax2.plot(x, np.mean(mse_train,axis=0),
             c='orange', linestyle='--',label = 'MSE train', linewidth=3, ms=1)
    ax2.scatter(best_num_m,min_mse,c='red',label = 'lowest test mse', s=100, marker='X')
    ax2.fill_between(x,
                     np.mean(mse_test,axis=0)-1.96*np.std(mse_test,axis=0),
                     np.mean(mse_test,axis=0)+1.96*np.std(mse_test,axis=0),
                    color='yellow', alpha=0.3, label='conf. bound test MSE')
    ax2.set_ylabel('MSE')
    fig.legend()
    plt.tight_layout()
    plt.show()
    print('Results:')
    print("Criterion\t\t\tNum.auto-regressors\t\tAdj. R2 test\t\tAIC test\tMSE test")
    print("-------------------------------------------------------------------------------------------------------------")
    print("%s:\t\t\t%1.0f\t\t\t\t%1.4f\t\t\t%1.4f\t\t%1.4f" % ('max adj. R2', best_num_r,
                                                               np.mean(r2_adj_test, axis=0)[best_num_r-1],
                                                               np.mean(aic_test, axis=0)[best_num_r-1],
                                                               np.mean(mse_test, axis=0)[best_num_r-1]))
    print("%s:\t\t\t%1.0f\t\t\t\t%1.4f\t\t\t%1.4f\t\t%1.4f" % ('min MSE', best_num_m,
                                                               np.mean(r2_adj_test, axis=0)[best_num_m-1],
                                                               np.mean(aic_test, axis=0)[best_num_m-1],
                                                               np.mean(mse_test, axis=0)[best_num_m-1]))
    print("%s:\t\t\t%1.0f\t\t\t\t%1.4f\t\t\t%1.4f\t\t%1.4f" % ('first in range', first_in_range,
                                                               np.mean(r2_adj_test, axis=0)[best-1],
                                                               np.mean(aic_test, axis=0)[best-1],
                                                               np.mean(mse_test, axis=0)[best-1]))
    print("-------------------------------------------------------------------------------------------------------------")
    return first_in_range, best_num_r, best_num_m, sorted_lags




# ------------------------ RFECV ------------------------
from sklearn.feature_selection import RFECV
def scorer_adj_r2(estimator, X, y):
    n = X.shape[0]
    p = X.shape[1]
    y_pred = estimator.predict(X)
    R2 = r2_score(y, y_pred)
    return 1-(1-R2)*(n-1)/(n-p-1)


def rfecv_selection(
        house, max_num_days, keep_non_ar,
        train_years, valid_years, step_ahead=1,
        repeats=5, verbose=True, plot_fig=True):
    data_power_org = copy.deepcopy(house.data_power)
    # build regression df with all lags
    house.data_power = _augment_lags(
                            house.data_power, lags=np.arange(1, max_num_days*24+1),
                            step_ahead=step_ahead)
    (X_train, y_train, _, _) = house.construct_regression_matrices(
                            m_train=None, train_years=train_years,
                            valid_years=valid_years)
    print(X_train.shape)
    house.data_power = data_power_org

    # set up rfecv
    min_features_to_select=1
    rfecv = RFECV(estimator=SMWrapper(sm.OLS, fit_intercept=True), step=1, cv=repeats,
                  scoring=scorer_adj_r2,
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X_train, y_train.flatten())

    #Force every non-lag feature to be in the RFECV
    if keep_non_ar:
        for i, feature_name in enumerate(house.feature_names):
            if not feature_name.startswith('lag'):
                rfecv.support_[i]=True

    # selected features
    feat_rfecv = [house.feature_names[i] for i, x in enumerate(rfecv.support_) if x]

    # Plot number of features VS. cross-validation scores
    if plot_fig:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation adj. R2 score")
        plt.plot(range(min_features_to_select,
                       len(rfecv.grid_scores_) + min_features_to_select),
                 rfecv.grid_scores_)
        plt.show()

    # display results
    if verbose:
        print("\nRecurssive Feature Elimination + CV")
        print("Optimal number of features: %d" % len(feat_rfecv))
        print("Optimal number of lags: %d" + str(len(feat_rfecv)-8))
        print("Selected features:")
        print(feat_rfecv)

    # divide
    lag_rfecv = np.array([int(x.replace('lag ','')) for x in feat_rfecv if x.startswith('lag ')])
    feat_col_rfecv = [x for x in feat_rfecv if not x.startswith('lag ') and not x=='constant']
    return lag_rfecv, feat_col_rfecv




if __name__ == "__main__":
    import os, sys
    from house_pv import HousePV
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(PROJECT_DIR)
    from Synthetic_PV_Profiles import WeatherStation

    # DATA FROM PVGIS USING TILT=0 i.e. IRRADIANCE ON NORMAL PLANE
    city_name = 'Lausanne'
    weather_station = WeatherStation(city_name)

    house = HousePV(tilt=0, azimuth=180,
                    module_name='Canadian_Solar_CS5P_220M___2009_',
                    inverter_name='ABB__MICRO_0_25_I_OUTD_US_208__208V_',
                    latitude=45.473, longitude=9.185, altitude=133,
                    weather_dev=0.1, irrad_scale=1,
                    weather_station=weather_station,
                    shadows={'st': 3, 'en': 8, 'peak_red':0.8},
                    random_state=None)

    house.simulate_pv()
    house.construct_regression_df(
                    use_station_irrad_direct=True, use_station_irrad_diffuse=True,
                    delay_irrad=True, lags=[1], hours=np.arange(7,18),
                    months=[3,4], step_ahead=1)
    (X_train, y_train, X_valid, y_valid) = house.construct_regression_matrices(
                    m_train=None, train_years=[2018, 2019],
                    valid_years=2020)
    #first_in_range, best_num_r, best_num_m, sorted_lags = tune_pacf(
    #                                house, max_num_lags=15, max_num_days=1,
    #                                step_ahead=1, repeats=3)
    #print('[RES] lags by first in range: ', *sorted_lags[:first_in_range])
    #print('[RES] lags by lowest adj r2: ', *sorted_lags[:best_num_r])

    lag_rfecv, feat_col_rfecv = rfecv_selection(
        house, max_num_days=7, keep_non_ar=False, step_ahead=1,
        repeats=5, verbose=True, plot_fig=True)
    print(lag_rfecv)
    print(feat_col_rfecv)