from itertools import repeat
from sklearn.model_selection import ParameterGrid
from botProto1 import *
import warnings
import argparse


warnings.filterwarnings("ignore",category =RuntimeWarning)

#os.system("taskset -p 0xff %d" % os.getpid())


class optimizer(object):

    def __init__(self,n_proc,frame,dates):

        self.n_proc = n_proc
        self.error_vals = [2.0]#,5.0,10.0,15.0,20.0]
        self.stop_vals = [1.0,1.5]#,2.0]
        self.peak_vals = [5]#,10,15,20]
        self.atrs = [5, 7]#, 10, 14, 21]
        self.results = pd.DataFrame(columns=['stop','peak','error','atr_range','sharpe','apr','acc','exp'])
        self.frame = frame
        self.dates = dates

    def prep(self):

        self.data = backtestData(frame=self.frame,n_split=500,pairs=['EUR_USD','GBP_USD','AUD_USD','NZD_USD'],dates=self.dates)
        self.bot = PatternBot(data=data,instrument=pairs,pairs=['EUR_USD','GBP_USD','AUD_USD','NZD_USD'])
        parameters = {'stop':self.stop_vals,'peak':self.peak_vals,'error':self.error_vals,'atrs':self.atrs}
        self.grid = ParameterGrid(parameters)

        stops = [d['stop'] for d in self.grid]
        peaks = [d['peak'] for d in self.grid]
        error = [d['error'] for d in self.grid]
        atrs = [d['atrs'] for d in self.grid]

        self.grid = list(zip(stops,peaks,error,atrs))


    def ret_func(self,retval):

        retval = retval[1]

        self.results = self.results.append({'stop':retval[0],'peak':retval[1],'error':retval[2],'atr_range':retval[3],'sharpe':retval[4],
                                            'apr':retval[5],'acc':retval[6],'exp':retval[7]},ignore_index=True)

    def search(self):

        self.start = time.time()

        p = multiprocessing.Pool(processes=self.n_proc)

        for x, y in zip(repeat(self.data),self.grid):

            r = p.apply_async(self.bot.backtest,(x,y),callback=self.ret_func)

        p.close()
        p.join()

        # Get Best Performance

        sharpe_idx = self.results.sharpe.idxmax()
        apr_idx = self.results.apr.idxmax()
        acc_idx = self.results.acc.idxmax()
        exp_idx = self.results.exp.idxmax()

        tmp = self.results[['stop','atr_range','peak','error']]

        performance = {'sharpe':[self.results.sharpe.max(),tmp.iloc[sharpe_idx].values],
                       'apr':[self.results.apr.max(),tmp.iloc[apr_idx].values],
                       'acc':[self.results.acc.max(),tmp.iloc[acc_idx].values],
                       'exp':[self.results.exp.max(),tmp.iloc[exp_idx].values]}

        return performance