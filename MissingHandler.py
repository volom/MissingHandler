import pandas as pd
import numpy as np
import statistics as stat

from sklearn.linear_model import LinearRegression


class MissingHandler:
   
    def __init__(self, x, y, method):
        self.x = x
        self.y = y
        self.method = method

    # functions to handle missing values
    def __missing_average_strategy(self, x):
        x = pd.DataFrame(x)
        for column in x.columns:
            x_var = x[column].values
            v_put = np.mean(x_var[~np.isnan(x_var)])
            x_var[np.isnan(x_var)] = v_put
            x[column] = x_var
        return x
    
    def __missing_median_strategy(self, x):
        x = pd.DataFrame(x)
        for column in x.columns:
            x_var = x[column].values
            v_put = stat.median(x_var[~np.isnan(x_var)])
            x_var[np.isnan(x_var)] = v_put
            x[column] = x_var
        return x
    
    def __missing_mode_strategy(self, x):
        x = pd.DataFrame(x)
        for column in x.columns:
            x_var = x[column].values
            v_put = stat.mode(x_var[~np.isnan(x_var)])
            x_var[np.isnan(x_var)] = v_put
            x[column] = x_var
        return x

    def __geo_mean(self, iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    def __missing_geomean_strategy(self, x):
        x = pd.DataFrame(x)
        for column in x.columns:
            x_var = x[column].values
            v_put = self.__geo_mean(x_var[~np.isnan(x_var)])
            x_var[np.isnan(x_var)] = v_put
            x[column] = x_var
        return x
    
    def __missing_delete_strategy(self, x, y):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        x = x.dropna(how='any')
        y = y.dropna(how='any')
        return x, y
    
    def __missing_min_strategy(self, x):
        x = pd.DataFrame(x)
        for column in x.columns:
            x_var = x[column].values
            v_put = min(x_var[~np.isnan(x_var)])
            x_var[np.isnan(x_var)] = v_put
            x[column] = x_var
        return x

    def __missing_max_strategy(self, x):
        x = pd.DataFrame(x)
        for column in x.columns:
            x_var = x[column].values
            v_put = max(x_var[~np.isnan(x_var)])
            x_var[np.isnan(x_var)] = v_put
            x[column] = x_var
        return x

    def __missing_last_observe_strategy(self, x):
        x = pd.DataFrame(x)

        for column in x.columns:
            x_var = x[column].values
            # previous value if nan index is not zero else next value

            # fill the next value for the first serial nans
            if np.isnan(x_var[0]):
                for i in map(str, list(x_var)):
                    if i != 'nan':
                        x_var[[i for i in range(list(map(str, list(x_var))).index(i))]] = i
                        break
            # fill na value with previous value in case of na sequence
            while np.count_nonzero(np.isnan(x_var)) != 0:
                temp_lst = list(filter(lambda i: list(np.isnan(x_var))[i], range(len(list(np.isnan(x_var))))))
                if len(temp_lst) != 0:
                    temp_lst = list(map(lambda x: x-1, temp_lst))

                    for index in temp_lst[::-1]:
                        x_var[index+1] = x_var[index]

            x[column] = x_var
            return x
        
    def __missing_next_observe_strategy(self, x):
        x = pd.DataFrame(x)

        for column in x.columns:
            x_var = x[column].values
            # fill the previous value for the last serial nans
            if np.isnan(x_var[-1]):
                for i in map(str, list(x_var[::-1])):
                    if i != 'nan':
                        x_var[list(map(str, list(x_var))).index(i)+1:] = i
                        break
            # fill na value with previous value in case of na sequence
            while np.count_nonzero(np.isnan(x_var)) != 0:
                temp_lst = list(filter(lambda i: list(np.isnan(x_var))[i], range(len(list(np.isnan(x_var))))))
                if len(temp_lst) != 0:
                    temp_lst = list(map(lambda x: x+1, temp_lst))

                    for index in temp_lst[::-1]:
                        x_var[index-1] = x_var[index]

            x[column] = x_var
            return x
        
    def __linear_strategy(self, x, y):
        df_xy = pd.concat([x, y], axis=1)
        df_xy.dropna(how='any', inplace=True)
        
        x_withna = pd.DataFrame(columns=x.columns)
        y_withna = pd.DataFrame(columns=y.columns)
        x_notna = df_xy[x.columns]
        y_notna = df_xy[y.columns]

        for column in x.columns:
            x_withna = pd.concat([x_withna, x[x[column].isna()==True]])
        y_withna = y.iloc[list(x_withna.index), :]
            
        regressor = LinearRegression()
        regressor.fit(x_notna.values, y_notna.values)

        a1 = [x for x in list(regressor.coef_)[0]]
        b1 =  regressor.intercept_[0]
        b2 = list(y_withna.iloc[:,0])

        x_withna.reset_index(drop=True, inplace=True)
        pd_index = 0
        none_index = 0
        
        while pd_index < len(x_withna):
            a2 = list(x_withna.iloc[pd_index, :])
            res = 0
            div = 1
            index = 0

            while index < len(a1):
                if str(a2[index]) != 'nan':
                    res += a1[index] * a2[index]
                else:
                    div = a1[index]
                    none_index = index
                index += 1

            res = b2[pd_index] - res - b1
            res = res / div

            x_withna.iloc[pd_index, none_index] = res
            pd_index += 1
        return pd.concat([x_withna, x_notna]), pd.concat([y_withna, y_notna])
    
    # transform funct to run handling functions
    def transform(self):
        if self.method == 'average':
            return self.__missing_average_strategy(self.x)
        elif self.method == 'median':
            return self.__missing_median_strategy(self.x)
        elif self.method == 'mode':
            return self.__missing_mode_strategy(self.x)
        elif self.method == 'geomean':
            return self.__missing_geomean_strategy(self.x)
        elif self.method == 'delete':
            return self.__missing_delete_strategy(self.x, self.y)
        elif self.method == 'min':
            return self.__missing_min_strategy(self.x)
        elif self.method == 'max':
            return self.__missing_max_strategy(self.x)                
        elif self.method == 'last_observe':
            return self.__missing_last_observe_strategy(self.x)           
        elif self.method == 'next_observe':
            return self.__missing_next_observe_strategy(self.x)  
        elif self.method == 'linear_strategy':
            return self.__linear_strategy(self.x, self.y)  
   