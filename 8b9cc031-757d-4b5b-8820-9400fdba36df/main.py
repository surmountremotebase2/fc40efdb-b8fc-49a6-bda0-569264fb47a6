#region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
import statsmodels.api as sm
import statistics as stat
import pickle
#endregion

class Q2PlaygroundAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2008, 1, 1)   # Set Start Date
        self.set_cash(100000)             # Set Strategy Cash
        self.set_security_initializer(BrokerageModelSecurityInitializer(
            self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)
        ))
        ########################## PARAMETERS ##########################
        # self.p_lookback = self.get_parameter("p_lookback", 252)
        # self.p_num_coarse = self.get_parameter("p_num_coarse", 200)
        # self.p_num_fine = self.get_parameter("p_num_fine", 70)
        # self.p_num_long = self.get_parameter("p_num_long", 5)
        # self.p_adjustment_step = self.get_parameter("p_adjustment_step", 1.0)
        # self.p_n_portfolios = self.get_parameter("p_n_portfolios", 1000)
        # self.p_short_lookback = self.get_parameter("p_short_lookback", 63)
        # self.p_rand_seed = self.get_parameter("p_rand_seed", 13)
        ################################################################
        self.p_lookback = 252
        self.p_num_coarse = 200
        self.p_num_fine = 70
        self.p_num_long = 5
        self.p_adjustment_step = 1.0
        self.p_n_portfolios = 1000
        self.p_short_lookback = 63
        self.p_rand_seed = 13
        self.p_adjustment_frequency = 'monthly'  # Can be 'monthly', 'weekly', 'bi-weekly'
        ################################################################
        self.universe_settings.resolution = Resolution.DAILY

        self._momp = {}          # Dict of Momentum indicator keyed by Symbol
        self._lookback = self.p_lookback     # Momentum indicator lookback period
        self._num_coarse = self.p_num_coarse # Number of symbols selected at Coarse Selection
        self._num_fine = self.p_num_fine     # Number of symbols selected at Fine Selection
        self._num_long = self.p_num_long     # Number of symbols with open positions

        self._rebalance = False
        self.current_holdings = set()  # To track current holdings

        self.target_weights = {}  # To store target weights
        self.adjustment_step = self.p_adjustment_step  # Adjustment step for gradual transition

        self.first_trade_date = None
        self.next_adjustment_date = None

        self.add_universe(self._coarse_selection_function, self._fine_selection_function)

    def _coarse_selection_function(self, coarse):
        '''Drop securities which have no fundamental data or have too low prices.
        Select those with highest by dollar volume'''
        if self.next_adjustment_date and self.time < self.next_adjustment_date:
            return Universe.UNCHANGED

        self._rebalance = True

        if not self.first_trade_date:
            self.first_trade_date = self.time
            self.next_adjustment_date = self.get_next_adjustment_date(self.time)
            self._rebalance = True

        selected = sorted([x for x in coarse if x.has_fundamental_data and x.price > 5],
            key=lambda x: x.dollar_volume, reverse=True)

        return [x.symbol for x in selected[:self._num_coarse]]

    def _fine_selection_function(self, fine):
        '''Select security with highest market cap'''
        selected = sorted(fine, key=lambda f: f.market_cap, reverse=True)
        return [x.symbol for x in selected[:self._num_fine]]

    def on_data(self, data):
        # Update the indicator
        for symbol, mom in self._momp.items():
            mom.update(self.time, self.securities[symbol].close)

        # Check if empty portfolio and set first_trade_date
        if not self.Portfolio.Invested and not self.first_trade_date:
            self.first_trade_date = self.time
            self.next_adjustment_date = self.get_next_adjustment_date(self.time, initial=True)
            self._rebalance = True

        if not self._rebalance:
            return

        # Selects the securities with highest momentum
        sorted_mom = sorted([k for k,v in self._momp.items() if v.is_ready],
            key=lambda x: self._momp[x].current.value, reverse=True)
        selected = sorted_mom[:self._num_long]
        new_holdings = set(selected)

        # Only rebalance if the new selection is different from current holdings
        if new_holdings != self.current_holdings or self.first_trade_date == self.time:
            if len(selected) > 0:
                optimal_weights = self.optimize_portfolio(selected)
                self.target_weights = dict(zip(selected, optimal_weights))
                self.current_holdings = new_holdings
                self.adjust_portfolio()

        self._rebalance = False
        self.next_adjustment_date = self.get_next_adjustment_date(self.time)

    def on_securities_changed(self, changes):
        # Clean up data for removed securities and Liquidate
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if self._momp.pop(symbol, None) is not None:
                self.Liquidate(symbol, 'Removed from universe')

        for security in changes.AddedSecurities:
            if security.Symbol not in self._momp:
                self._momp[security.Symbol] = MomentumPercent(self._lookback)

        # Warm up the indicator with history price if it is not ready
        added_symbols = [k for k, v in self._momp.items() if not v.IsReady]

        history = self.History(added_symbols, 1 + self._lookback, Resolution.Daily)
        history = history.close.unstack(level=0)

        for symbol in added_symbols:
            ticker = symbol.ID.ToString()
            if ticker in history:
                for time, value in history[ticker].dropna().items():
                    item = IndicatorDataPoint(symbol, time.date(), value)
                    self._momp[symbol].Update(item)

    def optimize_portfolio(self, selected_symbols):
        short_lookback = self.p_short_lookback
        returns = self.history(selected_symbols, short_lookback, Resolution.DAILY)['close'].unstack(level=0).pct_change().dropna()
        n_assets = len(selected_symbols)
        n_portfolios = self.p_n_portfolios

        results = np.zeros((3, n_portfolios))
        weights_record = []

        np.random.seed(self.p_rand_seed)

        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = np.sum(returns.mean() * weights) * short_lookback
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * short_lookback, weights)))

            downside_stddev = np.sqrt(np.mean(np.minimum(0, returns).apply(lambda x: x**2, axis=0).dot(weights)))
            sortino_ratio = portfolio_return / downside_stddev

            results[0,i] = portfolio_return
            results[1,i] = portfolio_stddev
            results[2,i] = sortino_ratio

            weights_record.append(weights)

        best_sortino_idx = np.argmax(results[2])
        return weights_record[best_sortino_idx]

    def adjust_portfolio(self):
        current_symbols = set(self.Portfolio.Keys)
        target_symbols = set(self.target_weights.keys())

        # Liquidate removed symbols
        removed_symbols = current_symbols - target_symbols
        for symbol in removed_symbols:
            self.Liquidate(symbol)

        # Adjust holdings for selected symbols
        for symbol, target_weight in self.target_weights.items():
            current_weight = self.Portfolio[symbol].Quantity / self.Portfolio.TotalPortfolioValue if symbol in self.Portfolio else 0
            adjusted_weight = current_weight * (1 - self.adjustment_step) + target_weight * self.adjustment_step
            self.SetHoldings(symbol, adjusted_weight)

    def get_next_adjustment_date(self, current_date, initial=False):
        if self.p_adjustment_frequency == 'weekly':
            return current_date + timedelta(days=7)
        elif self.p_adjustment_frequency == 'bi-weekly':
            return current_date + timedelta(days=14)
        elif self.p_adjustment_frequency == 'monthly':
            if initial:
                next_month = current_date.replace(day=1) + timedelta(days=32)
                return next_month.replace(day=1)
            next_month = current_date.replace(day=1) + timedelta(days=32)
            return next_month.replace(day=1)
        else:
            raise ValueError(f"Unsupported adjustment frequency: {self.p_adjustment_frequency}")
