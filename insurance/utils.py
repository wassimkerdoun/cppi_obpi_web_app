import os
import random
import string
import numpy as np
import yfinance as yf
import datetime
from arch import arch_model
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.dates as mdates
import pandas as pd
import logging

class PortfolioInsurance:
    def __init__(self, initial_capital, floor, risky_ticker, start_date, end_date=None):
        self.initial_capital = initial_capital
        self.floor = floor * initial_capital
        self.risky_ticker = risky_ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.datetime.now().strftime('%Y-%m-%d')
        self.jump_lambda = 1
        self.jump_mu =  0.1
        self.jump_sigma = 0.15
        self.download_data()
        self.calculate_volatility()
        
    def download_data(self):
        print(f"Downloading data for {self.risky_ticker}...")
        risky_data = yf.download(self.risky_ticker, start=self.start_date, end=self.end_date)
        
        self.risk_asset_returns = risky_data['Close'].pct_change().dropna().values
        self.risky_prices = risky_data['Close'].values
        
        min_length = len(self.risk_asset_returns)
        self.risk_asset_returns = self.risk_asset_returns[-min_length:]
        self.risky_prices = self.risky_prices[-min_length:]
        
        self.num_periods = min_length
        self.dates = risky_data.index[-min_length:]
        
    def calculate_volatility(self):
        model = arch_model(self.risk_asset_returns * 100, vol='GARCH', p=1, q=1)
        res = model.fit(disp='off')
        self.conditional_volatility = res.conditional_volatility / 100
        self.annualized_volatility = self.conditional_volatility * np.sqrt(252)
        
    def black_scholes(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    def black_scholes_delta(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)
        
    def merton_jump_diffusion(self, S, K, T, r, sigma, option_type="call", 
                            n_paths=10000, n_steps=1):
        
        # Nombre de sauts pour chaque trajectoire
        N = np.random.poisson(self.jump_lambda * T, size=n_paths)

        # Sauts agrégés pour chaque trajectoire
        jump_sum = np.random.normal(
            loc=N * self.jump_mu,
            scale=np.sqrt(N) * self.jump_sigma
        )

        # Partie diffusion (sans sauts)
        Z = np.random.normal(0, 1, size=n_paths)
        diffusion = (r - 0.5 * sigma**2 - self.jump_lambda * 
                    (np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1)) * T \
                    + sigma * np.sqrt(T) * Z

        log_ST = np.log(S) + diffusion + jump_sum
        ST = np.exp(log_ST)

        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price


    def calculate_max_drawdown(self, values):
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max
        return np.max(drawdowns)
    
    def calculate_sharpe_ratio(self, returns, annual_risk_free_rate=0.02):
        daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_risk_free_rate
        mean_excess_returns = np.mean(excess_returns)
        std_excess_returns = np.std(excess_returns)
        if std_excess_returns == 0:
            return np.nan
        daily_sharpe = mean_excess_returns / std_excess_returns
        annual_sharpe = daily_sharpe * np.sqrt(252)
        return annual_sharpe
        
    def run_obpi_bond_call(self, risk_free_rate=0.02, merton='merton jump'):
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        T_total = (end_dt - start_dt).days / 365.25
        S0 = self.risky_prices[0]
        
        # Bond investment to guarantee floor at maturity
        bond_investment = self.floor * np.exp(-risk_free_rate * T_total)
        remaining_capital = self.initial_capital - bond_investment
        K = S0

        if merton=='merton jump':
            call_premium = self.merton_jump_diffusion(
                S=S0,
                K=K,
                T=T_total,
                r=risk_free_rate,
                sigma=self.annualized_volatility[0],
                option_type="call"
            )
        else:
            call_premium = self.black_scholes(
                S=S0,
                K=K,
                T=T_total,
                r=risk_free_rate,
                sigma=self.annualized_volatility[0],
                option_type="call"
            )

        n_calls = remaining_capital / call_premium

        portfolio_values = np.zeros(self.num_periods)
        bond_values = np.zeros(self.num_periods)
        call_values = np.zeros(self.num_periods)
        buy_and_hold_values = (self.initial_capital / S0) * self.risky_prices

        for t in range(self.num_periods):
            days_passed = (self.dates[t] - start_dt).days
            time_remaining = max(T_total - days_passed/365.25, 1e-5)
            St = self.risky_prices[t]
            bond_values[t] = bond_investment * np.exp(risk_free_rate * days_passed/365.25)

            if merton=='merton jump':
                call_price = self.merton_jump_diffusion(
                    S=St,
                    K=K,
                    T=time_remaining,
                    r=risk_free_rate,
                    sigma=self.annualized_volatility[t],
                    option_type="call"
                )
            else:
                call_price = self.black_scholes(
                    S=St,
                    K=K,
                    T=time_remaining,
                    r=risk_free_rate,
                    sigma=self.annualized_volatility[t],
                    option_type="call"
                )
            call_values[t] = n_calls * call_price
            portfolio_values[t] = bond_values[t] + call_values[t]

            
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        buy_and_hold_returns = buy_and_hold_values[1:] / buy_and_hold_values[:-1] - 1
        
        mdd_portfolio = self.calculate_max_drawdown(portfolio_values)
        mdd_buy_and_hold = self.calculate_max_drawdown(buy_and_hold_values)
        sharpe_portfolio = self.calculate_sharpe_ratio(portfolio_returns)
        sharpe_buy_and_hold = self.calculate_sharpe_ratio(buy_and_hold_returns)

        return {
            'dates': self.dates,
            'portfolio_values': portfolio_values,
            'bond_values': bond_values,
            'call_values': call_values,
            'buy_and_hold_values': buy_and_hold_values,
            'volatility': self.annualized_volatility,
            'mdd_portfolio': mdd_portfolio,
            'mdd_buy_and_hold': mdd_buy_and_hold,
            'sharpe_portfolio': sharpe_portfolio,
            'sharpe_buy_and_hold': sharpe_buy_and_hold,
            'n_shares': n_calls.item() if isinstance(n_calls, np.ndarray) else n_calls
        }

        
    def run_obpi_underlying_put(self, risk_free_rate=0.02, merton='merton jump'):
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        T_total = (end_dt - start_dt).days / 365.25
        S0 = self.risky_prices[0]
        
        if merton=='merton jump':
            put_price = self.merton_jump_diffusion(
                S=S0,
                K=S0,
                T=T_total,
                r=risk_free_rate,
                sigma=self.annualized_volatility[0],
                option_type="put"
            )
        else:
            put_price = self.black_scholes(
                S=S0,
                K=S0,
                T=T_total,
                r=risk_free_rate,
                sigma=self.annualized_volatility[0],
                option_type="put"
            )
        
        total_cost_per_unit = S0 + put_price
        n_shares = self.initial_capital / total_cost_per_unit
        
        portfolio_values = np.zeros(self.num_periods)
        put_values = np.zeros(self.num_periods)
        risky_asset_values = np.zeros(self.num_periods)
        buy_and_hold_values = (self.initial_capital / S0) * self.risky_prices
        
        for t in range(self.num_periods):
            days_passed = (self.dates[t] - start_dt).days
            time_remaining = max(T_total - days_passed/365.25, 1e-5)
            St = self.risky_prices[t]
            
            risky_asset_values[t] = n_shares * St
            
            if merton=='merton jump':
                current_put_price = self.merton_jump_diffusion(
                    S=St,
                    K=S0,
                    T=time_remaining,
                    r=risk_free_rate,
                    sigma=self.annualized_volatility[min(t, len(self.annualized_volatility)-1)],
                    option_type="put"
                )
            else:
                current_put_price = self.black_scholes(
                    S=St,
                    K=S0,
                    T=time_remaining,
                    r=risk_free_rate,
                    sigma=self.annualized_volatility[min(t, len(self.annualized_volatility)-1)],
                    option_type="put"
                )
            put_values[t] = n_shares * current_put_price
            portfolio_values[t] = risky_asset_values[t] + put_values[t]

            
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        buy_and_hold_returns = buy_and_hold_values[1:] / buy_and_hold_values[:-1] - 1        
        mdd_portfolio = self.calculate_max_drawdown(portfolio_values)
        mdd_buy_and_hold = self.calculate_max_drawdown(buy_and_hold_values)
        sharpe_portfolio = self.calculate_sharpe_ratio(portfolio_returns)
        sharpe_buy_and_hold = self.calculate_sharpe_ratio(buy_and_hold_returns)
        
        return {
            'dates': self.dates,
            'portfolio_values': portfolio_values,
            'put_values': put_values,
            'risky_asset_values': risky_asset_values,
            'buy_and_hold_values': buy_and_hold_values,
            'volatility': self.annualized_volatility,
            'mdd_portfolio': mdd_portfolio,
            'mdd_buy_and_hold': mdd_buy_and_hold,
            'sharpe_portfolio': sharpe_portfolio,
            'sharpe_buy_and_hold': sharpe_buy_and_hold,
            'n_shares': n_shares.item() if isinstance(n_shares, np.ndarray) else n_shares
        }

        
    def plot_strategy_comparison(self, results_call_bond, results_put_stock, base_filename, base_path):
        suffix = self.random_suffix()
        filenames = {}

        try:
            # Ensure that the dates are in datetime format
            results_call_bond['dates'] = pd.to_datetime(results_call_bond['dates'])
            results_put_stock['dates'] = pd.to_datetime(results_put_stock['dates'])

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(results_call_bond['dates'], results_call_bond['portfolio_values'], label='Call + Bond', color='#1f77b4')
            ax.plot(results_put_stock['dates'], results_put_stock['portfolio_values'], label='Stock + Put', color='#ff7f0e')
            ax.plot(results_call_bond['dates'], results_call_bond['buy_and_hold_values'], label='Buy-and-Hold', linestyle='--', color='#2ca02c')
            ax.axhline(self.floor, color='r', linestyle=':', label='Floor')

            ax.set_title('Portfolio Value: Call+Bond vs Stock+Put')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            # Generate file path
            comparison_filename = f"{base_filename}_strategy_comparison_{suffix}.png"
            comparison_path = os.path.join(base_path, comparison_filename)

            os.makedirs(base_path, exist_ok=True)

            fig.savefig(comparison_path, bbox_inches='tight')
            plt.close(fig)

            filenames['strategy_comparison'] = comparison_filename
            logging.info(f"Generated plot saved at {comparison_path}")
            return filenames

        except Exception as e:
            logging.error(f"Error generating plot: {str(e)}", exc_info=True)
            return filenames

    # Function to generate random suffix
    def random_suffix(self, length=6):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def _plot_generic(self, results, strategy_type, base_filename, base_path):
        suffix = self.random_suffix()
        filenames = {}

        # Plot 1: Portfolio value
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(results['dates'], results['portfolio_values'], label='Strategy', color='#1f77b4')
        ax1.plot(results['dates'], results['buy_and_hold_values'], label='Buy-and-Hold', linestyle='--', color='#ff7f0e')
        ax1.axhline(self.floor, color='r', linestyle=':', label='Floor')
        ax1.set_title('Portfolio Value Comparison')
        ax1.legend()
        portfolio_filename = f"{base_filename}_portfolio_{suffix}.png"
        portfolio_path = os.path.join(base_path, portfolio_filename)
        fig.savefig(portfolio_path, bbox_inches='tight')
        plt.close(fig)
        filenames['portfolio'] = portfolio_filename

        # Plot 2: Component breakdown
        fig, ax2 = plt.subplots(figsize=(10, 5))
        if strategy_type == 'put':
            ax2.plot(results['dates'], results['risky_asset_values'], label='Stock Value', color='#2ca02c')
            ax2.plot(results['dates'], results['put_values'], label='Put Options', color='#d62728')
        elif strategy_type == 'call':
            ax2.plot(results['dates'], results['bond_values'], label='Bond Value', color='#2ca02c')
            ax2.plot(results['dates'], results['call_values'], label='Call Options', color='#d62728')
        ax2.set_title('Strategy Components')
        ax2.legend()
        components_filename = f"{base_filename}_components_{suffix}.png"
        components_path = os.path.join(base_path, components_filename)
        fig.savefig(components_path, bbox_inches='tight')
        plt.close(fig)
        filenames['components'] = components_filename

        # Plot 3: Volatility
        fig, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(results['dates'], results['volatility'] * 100, label='Annualized Volatility', color='#9467bd')
        ax3.set_title('Volatility of Risky Asset')
        ax3.set_ylabel('Volatility (%)')
        ax3.legend()
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        volatility_filename = f"{base_filename}_volatility_{suffix}.png"
        volatility_path = os.path.join(base_path, volatility_filename)
        fig.savefig(volatility_path, bbox_inches='tight')
        plt.close(fig)
        filenames['volatility'] = volatility_filename

        return filenames
    
    
    def run_cppi(self, risk_free_rate=0.02, multiplier=3, rebalance_frequency=1, transaction_cost=0):
        """
        Implements the Constant Proportion Portfolio Insurance (CPPI) strategy.
        
        Parameters:
        - risk_free_rate: Annual risk-free rate (default: 2%)
        - multiplier: Risk multiplier determining exposure to risky asset (default: 3)
        - rebalance_frequency: Rebalance every n days (default: 1 day)
        - transaction_cost: Cost per transaction as a proportion (default: 0.05%)
        
        Returns:
        - Dictionary containing strategy results including portfolio values, component values,
        and performance metrics.
        """
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(self.start_date)
        
        # Initialize arrays to store results
        portfolio_values = np.zeros(self.num_periods)
        risky_allocations = np.zeros(self.num_periods)
        risk_free_allocations = np.zeros(self.num_periods)
        cushions = np.zeros(self.num_periods)
        floors = np.zeros(self.num_periods)
        
        # Initial allocation
        portfolio_value = self.initial_capital
        floor_value = self.floor
        cushion = portfolio_value - floor_value
        exposure = multiplier * cushion
        
        # Constrain exposure to be between 0 and portfolio value
        exposure = max(0, min(exposure, portfolio_value))
        
        risky_allocation = exposure
        risk_free_allocation = portfolio_value - risky_allocation
        
        # Number of units of risky asset
        risky_units = risky_allocation / self.risky_prices[0] if self.risky_prices[0] > 0 else 0
        
        # Transaction costs for initial allocation
        transaction_cost_amount = risky_allocation * transaction_cost
        portfolio_value -= transaction_cost_amount
        risk_free_allocation -= transaction_cost_amount
        
        # Buy and hold for comparison
        buy_and_hold_units = self.initial_capital / self.risky_prices[0]
        buy_and_hold_values = buy_and_hold_units * self.risky_prices
        
        # Daily risk-free rate
        daily_rf_rate = ((1 + risk_free_rate) ** (1/252)) - 1
        
        # Track last rebalance day
        last_rebalance_day = 0
        
        # Run simulation
        for t in range(self.num_periods):
            # Update floor value (grows at risk-free rate)
            if t > 0:
                days_passed = (self.dates[t] - self.dates[t-1]).days
                floor_value *= (1 + daily_rf_rate) ** days_passed
            
            # Value of risky asset
            risky_value = risky_units * self.risky_prices[t]
            
            # Update risk-free asset value (grows at risk-free rate)
            if t > 0:
                days_passed = (self.dates[t] - self.dates[t-1]).days
                risk_free_allocation *= (1 + daily_rf_rate) ** days_passed
            
            # Current portfolio value
            portfolio_value = risky_value + risk_free_allocation
            
            # Current cushion
            cushion = portfolio_value - floor_value
            
            # Rebalance if needed (check if it's time to rebalance)
            if t - last_rebalance_day >= rebalance_frequency:
                # Calculate target exposure
                target_exposure = multiplier * cushion
                
                # Constrain exposure to be between 0 and portfolio value
                target_exposure = max(0, min(target_exposure, portfolio_value))
                
                # Current exposure
                current_exposure = risky_value
                
                # Calculate required adjustment
                adjustment = target_exposure - current_exposure
                
                if abs(adjustment) > 0.01 * portfolio_value:  # Only rebalance if adjustment is significant (> 1%)
                    # Calculate transaction cost
                    transaction_cost_amount = abs(adjustment) * transaction_cost
                    
                    # Update portfolio value after transaction cost
                    portfolio_value -= transaction_cost_amount
                    
                    # Recalculate cushion
                    cushion = portfolio_value - floor_value
                    
                    # Recalculate target exposure with updated cushion
                    target_exposure = multiplier * cushion
                    target_exposure = max(0, min(target_exposure, portfolio_value))
                    
                    # Update allocations
                    if adjustment > 0:  # Buy more risky asset
                        additional_units = adjustment / self.risky_prices[t]
                        risky_units += additional_units
                        risk_free_allocation -= (adjustment + transaction_cost_amount)
                    else:  # Sell risky asset
                        reduced_units = abs(adjustment) / self.risky_prices[t]
                        risky_units -= reduced_units
                        risk_free_allocation += (abs(adjustment) - transaction_cost_amount)
                    
                    # Update last rebalance day
                    last_rebalance_day = t
            
            # Record values
            portfolio_values[t] = portfolio_value
            risky_allocations[t] = risky_units * self.risky_prices[t]
            risk_free_allocations[t] = risk_free_allocation
            cushions[t] = cushion
            floors[t] = floor_value
        
        # Calculate strategy returns
        portfolio_returns = np.zeros(self.num_periods - 1)
        for i in range(1, self.num_periods):
            portfolio_returns[i-1] = (portfolio_values[i] / portfolio_values[i-1]) - 1
        
        # Calculate buy and hold returns
        buy_and_hold_returns = buy_and_hold_values[1:] / buy_and_hold_values[:-1] - 1
        
        # Calculate performance metrics
        mdd_portfolio = self.calculate_max_drawdown(portfolio_values)
        mdd_buy_and_hold = self.calculate_max_drawdown(buy_and_hold_values)
        sharpe_portfolio = self.calculate_sharpe_ratio(portfolio_returns, annual_risk_free_rate=risk_free_rate)
        sharpe_buy_and_hold = self.calculate_sharpe_ratio(buy_and_hold_returns, annual_risk_free_rate=risk_free_rate)
        
        return {
            'dates': self.dates,
            'portfolio_values': portfolio_values,
            'risky_allocations': risky_allocations,
            'risk_free_allocations': risk_free_allocations,
            'buy_and_hold_values': buy_and_hold_values,
            'volatility': self.annualized_volatility,
            'cushions': cushions,
            'floors': floors,
            'mdd_portfolio': mdd_portfolio,
            'mdd_buy_and_hold': mdd_buy_and_hold,
            'sharpe_portfolio': sharpe_portfolio,
            'sharpe_buy_and_hold': sharpe_buy_and_hold,
            'multiplier': multiplier
        }

    def plot_cppi_strategy(self, results, base_filename, base_path):
        """
        Plots the results of a CPPI strategy.
        
        Parameters:
        - results: Dictionary of results from run_cppi()
        - base_filename: Base filename for saving plots
        - base_path: Directory to save plots
        
        Returns:
        - Dictionary of generated filenames
        """
        suffix = self.random_suffix()
        filenames = {}
        
        try:
            # Ensure that the directory exists
            os.makedirs(base_path, exist_ok=True)
            
            # Ensure that the dates are in datetime format
            results['dates'] = pd.to_datetime(results['dates'])
            
            # Plot 1: Portfolio Value Comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(results['dates'], results['portfolio_values'], label='CPPI Strategy', color='#1f77b4')
            ax.plot(results['dates'], results['buy_and_hold_values'], label='Buy-and-Hold', linestyle='--', color='#ff7f0e')
            ax.plot(results['dates'], results['floors'], label='Floor', color='r', linestyle=':')
            
            ax.set_title(f'CPPI Strategy (Multiplier: {results["multiplier"]})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            portfolio_filename = f"{base_filename}_cppi_portfolio_{suffix}.png"
            portfolio_path = os.path.join(base_path, portfolio_filename)
            fig.savefig(portfolio_path, bbox_inches='tight')
            plt.close(fig)
            filenames['portfolio'] = portfolio_filename
            
            # Plot 2: Asset Allocation
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stackplot(results['dates'], 
                        [results['risky_allocations'], results['risk_free_allocations']], 
                        labels=['Risky Asset', 'Risk-Free Asset'],
                        colors=['#2ca02c', '#d62728'])
            
            ax.set_title('CPPI Asset Allocation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Allocation Value')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            allocation_filename = f"{base_filename}_cppi_allocation_{suffix}.png"
            allocation_path = os.path.join(base_path, allocation_filename)
            fig.savefig(allocation_path, bbox_inches='tight')
            plt.close(fig)
            filenames['allocation'] = allocation_filename
            
            # Plot 3: Cushion and Floor
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(results['dates'], results['cushions'], label='Cushion', color='#9467bd')
            ax.plot(results['dates'], results['floors'], label='Floor', color='r', linestyle=':')
            
            ax.set_title('CPPI Cushion and Floor Values')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            cushion_filename = f"{base_filename}_cppi_cushion_{suffix}.png"
            cushion_path = os.path.join(base_path, cushion_filename)
            fig.savefig(cushion_path, bbox_inches='tight')
            plt.close(fig)
            filenames['cushion'] = cushion_filename
            
            return filenames
            
        except Exception as e:
            logging.error(f"Error generating CPPI plots: {str(e)}", exc_info=True)
            return filenames
        
    def compare_all_strategies(self, risk_free_rate=0.02, cppi_multiplier=3, rebalance_frequency=1,
                               base_filename="strategy_comparison", base_path="plots"):
        """
        Runs and compares all portfolio insurance strategies: OBPI with call+bond, OBPI with stock+put, and CPPI.
        
        Parameters:
        - risk_free_rate: Annual risk-free rate (default: 2%)
        - cppi_multiplier: CPPI risk multiplier (default: 3)
        - base_filename: Base filename for saving plots
        - base_path: Directory to save plots
        
        Returns:
        - Dictionary of performance metrics and generated plot filenames
        """
        # Run the three strategies
        results_call_bond = self.run_obpi_bond_call(risk_free_rate=risk_free_rate)
        results_put_stock = self.run_obpi_underlying_put(risk_free_rate=risk_free_rate)
        results_cppi = self.run_cppi(risk_free_rate=risk_free_rate, multiplier=cppi_multiplier, rebalance_frequency=rebalance_frequency)
        
        # Create a comparison dataframe for performance metrics
        performance_metrics = pd.DataFrame({
            'Strategy': ['OBPI Call+Bond', 'OBPI Stock+Put', 'CPPI', 'Buy-and-Hold'],
            'Final Value': [
                results_call_bond['portfolio_values'][-1],
                results_put_stock['portfolio_values'][-1],
                results_cppi['portfolio_values'][-1],
                results_call_bond['buy_and_hold_values'][-1]  # Same for all strategies
            ],
            'Max Drawdown (%)': [
                results_call_bond['mdd_portfolio'] * 100,
                results_put_stock['mdd_portfolio'] * 100,
                results_cppi['mdd_portfolio'] * 100,
                results_call_bond['mdd_buy_and_hold'] * 100  # Same for all strategies
            ],
            'Sharpe Ratio': [
                results_call_bond['sharpe_portfolio'],
                results_put_stock['sharpe_portfolio'],
                results_cppi['sharpe_portfolio'],
                results_call_bond['sharpe_buy_and_hold']  # Same for all strategies
            ]
        })
        
        # Calculate total return
        initial_value = self.initial_capital
        performance_metrics['Total Return (%)'] = (performance_metrics['Final Value'] / initial_value - 1) * 100
        
        # Plot comparison
        filenames = self.plot_strategy_comparison_all(
            results_call_bond,
            results_put_stock,
            results_cppi,
            base_filename, 
            base_path
        )
        
        return {
            'performance_metrics': performance_metrics,
            'plot_filenames': filenames
        }

    def plot_strategy_comparison_all(self, results_call_bond, results_put_stock, results_cppi, base_filename, base_path):
        """
        Creates comparative plots for all three portfolio insurance strategies.
        
        Parameters:
        - results_call_bond: Results from run_obpi_bond_call()
        - results_put_stock: Results from run_obpi_underlying_put()
        - results_cppi: Results from run_cppi()
        - base_filename: Base filename for saving plots
        - base_path: Directory to save plots
        
        Returns:
        - Dictionary of generated filenames
        """
        suffix = self.random_suffix()
        filenames = {}

        try:
            # Ensure that the directory exists
            os.makedirs(base_path, exist_ok=True)
            
            # Ensure that the dates are in datetime format
            results_call_bond['dates'] = pd.to_datetime(results_call_bond['dates'])
            results_put_stock['dates'] = pd.to_datetime(results_put_stock['dates'])
            results_cppi['dates'] = pd.to_datetime(results_cppi['dates'])

            # Plot 1: Portfolio Values Comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_call_bond['dates'], results_call_bond['portfolio_values'], label='OBPI Call+Bond', color='#1f77b4')
            ax.plot(results_put_stock['dates'], results_put_stock['portfolio_values'], label='OBPI Stock+Put', color='#ff7f0e')
            ax.plot(results_cppi['dates'], results_cppi['portfolio_values'], label=f'CPPI (M={results_cppi["multiplier"]})', color='#2ca02c')
            ax.plot(results_call_bond['dates'], results_call_bond['buy_and_hold_values'], label='Buy-and-Hold', linestyle='--', color='#7f7f7f')
            ax.axhline(self.floor, color='r', linestyle=':', label='Floor')

            ax.set_title('Portfolio Insurance Strategies Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            comparison_filename = f"{base_filename}_all_strategies_{suffix}.png"
            comparison_path = os.path.join(base_path, comparison_filename)
            fig.savefig(comparison_path, bbox_inches='tight')
            plt.close(fig)
            filenames['all_strategies'] = comparison_filename

            # Plot 2: Risk-Adjusted Returns (Cumulative Return / Max Drawdown)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate cumulative returns
            call_bond_returns = results_call_bond['portfolio_values'] / results_call_bond['portfolio_values'][0] - 1
            put_stock_returns = results_put_stock['portfolio_values'] / results_put_stock['portfolio_values'][0] - 1
            cppi_returns = results_cppi['portfolio_values'] / results_cppi['portfolio_values'][0] - 1
            buy_hold_returns = results_call_bond['buy_and_hold_values'] / results_call_bond['buy_and_hold_values'][0] - 1
            
            ax.plot(results_call_bond['dates'], call_bond_returns * 100, label='OBPI Call+Bond', color='#1f77b4')
            ax.plot(results_put_stock['dates'], put_stock_returns * 100, label='OBPI Stock+Put', color='#ff7f0e')
            ax.plot(results_cppi['dates'], cppi_returns * 100, label=f'CPPI (M={results_cppi["multiplier"]})', color='#2ca02c')
            ax.plot(results_call_bond['dates'], buy_hold_returns * 100, label='Buy-and-Hold', linestyle='--', color='#7f7f7f')

            ax.set_title('Cumulative Returns Comparison')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return (%)')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            returns_filename = f"{base_filename}_cumulative_returns_{suffix}.png"
            returns_path = os.path.join(base_path, returns_filename)
            fig.savefig(returns_path, bbox_inches='tight')
            plt.close(fig)
            filenames['cumulative_returns'] = returns_filename
            
            return filenames

        except Exception as e:
            logging.error(f"Error generating plot: {str(e)}", exc_info=True)
            return filenames
        
        
        
# def main():
#     # Initialize portfolio insurance
#     initial_capital = 100000  # $100,000
#     floor_percentage = 0.85   # 85% of initial capital as floor
#     risky_ticker = "SPY"      # S&P 500 ETF
#     start_date = "2020-01-01"
#     end_date = "2022-12-31"   # Including COVID-19 crash and recovery
    
#     insurance = PortfolioInsurance(
#         initial_capital=initial_capital,
#         floor=floor_percentage,
#         risky_ticker=risky_ticker,
#         start_date=start_date,
#         end_date=end_date
#     )
    
#     # Configure and run strategies
#     risk_free_rate = 0.02  # 2% annual risk-free rate
#     cppi_multiplier = 3    # CPPI risk multiplier
    
#     # Run individual strategies
#     print("Running OBPI (Bond + Call) strategy...")
#     results_call_bond = insurance.run_obpi_bond_call(risk_free_rate=risk_free_rate)
    
#     print("Running OBPI (Stock + Put) strategy...")
#     results_put_stock = insurance.run_obpi_underlying_put(risk_free_rate=risk_free_rate)
    
#     print("Running CPPI strategy...")
#     results_cppi = insurance.run_cppi(
#         risk_free_rate=risk_free_rate,
#         multiplier=cppi_multiplier,
#         rebalance_frequency=5,  # Rebalance every 5 days
#         transaction_cost=0.0005  # 0.05% transaction cost
#     )
    
#     # Create output directory
#     output_dir = "portfolio_insurance_plots"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate plots for individual strategies
#     print("Generating plots for OBPI (Bond + Call)...")
#     insurance._plot_generic(results_call_bond, "call", "obpi_call_bond", output_dir)
    
#     print("Generating plots for OBPI (Stock + Put)...")
#     insurance._plot_generic(results_put_stock, "put", "obpi_stock_put", output_dir)
    
#     print("Generating plots for CPPI...")
#     insurance.plot_cppi_strategy(results_cppi, "cppi", output_dir)
    
#     # Compare all strategies
#     print("Comparing all strategies...")
#     comparison_results = insurance.compare_all_strategies(
#         risk_free_rate=risk_free_rate,
#         cppi_multiplier=cppi_multiplier,
#         base_filename="all_strategies",
#         base_path=output_dir
#     )
    
#     # Print performance metrics
#     print("\nPerformance Metrics:")
#     print(comparison_results['performance_metrics'].to_string(index=False))
    
#     # Print plot file locations
#     print("\nGenerated plot files:")
#     for plot_type, filename in comparison_results['plot_filenames'].items():
#         print(f"- {plot_type}: {os.path.join(output_dir, filename)}")
    
#     # Run sensitivity analysis for CPPI multiplier
#     print("\nRunning CPPI multiplier sensitivity analysis...")
#     multipliers = [1, 2, 3, 4, 5]
#     cppi_results = []
    
#     for m in multipliers:
#         print(f"  Testing multiplier {m}...")
#         result = insurance.run_cppi(risk_free_rate=risk_free_rate, multiplier=m)
#         cppi_results.append({
#             'Multiplier': m,
#             'Final Value': result['portfolio_values'][-1],
#             'Max Drawdown (%)': result['mdd_portfolio'] * 100,
#             'Sharpe Ratio': result['sharpe_portfolio']
#         })
    
#     # Create sensitivity analysis dataframe
#     sensitivity_df = pd.DataFrame(cppi_results)
#     sensitivity_df['Total Return (%)'] = (sensitivity_df['Final Value'] / initial_capital - 1) * 100
    
#     print("\nCPPI Multiplier Sensitivity Analysis:")
#     print(sensitivity_df.to_string(index=False))
    
#     print("\nPortfolio insurance analysis complete!")

# if __name__ == "__main__":
#     main()