import os
import logging
from datetime import datetime
from flask import Flask, render_template, session, request, redirect, url_for, flash
from insurance.utils import PortfolioInsurance

app = Flask(__name__)
app.secret_key = '13a9aad9f23bc281ca3aa07c'

# ---------------------------
# Basic index page for navigation
# ---------------------------
@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    session.clear()
    return render_template('index.html')



# ---------------------------
# OBPI form validation and data saving functions
# ---------------------------
def validate_form_data_obpi(form_data):
    required_fields = {
        'Initial Capital': 'initial_capital',
        'Floor': 'floor',
        'Risky Ticker': 'risky_ticker',
        'Start Date': 'start_date',
        'End Date': 'end_date',
        'Risk Free Rate': 'risk_free_rate',
        'Pricing Method': 'pricing_method',  # Black-Scholes or Merton Jump
        'Strategy': 'strategy'  # call+bond or put+underlying asset
    }
    
    for field, name in required_fields.items():
        if not form_data.get(name):
            return f"{field} is required"

    try:
        initial_capital = float(form_data.get('initial_capital'))
        floor = float(form_data.get('floor'))
        risk_free_rate = float(form_data.get('risk_free_rate'))
        if floor < 0 or floor > 1:
            return "Floor must be between 0 and 1"
        if initial_capital <= 0:
            return "Initial Capital must be positive"
        if risk_free_rate < 0:
            return "Risk Free Rate must be non-negative"
    except ValueError:
        return "All numeric fields must contain valid numbers"
    
    return None

def save_form_data_obpi(form_data):
    session['initial_capital'] = float(form_data.get('initial_capital', 10000))
    session['floor'] = float(form_data.get('floor', 0.9))
    session['risky_ticker'] = form_data.get('risky_ticker')
    session['start_date'] = form_data.get('start_date')
    session['end_date'] = form_data.get('end_date')
    session['risk_free_rate'] = float(form_data.get('risk_free_rate', 0.02))
    session['pricing_method'] = form_data.get('pricing_method')
    session['strategy'] = form_data.get('strategy')
    
def comparison_validate_form_data_obpi(form_data):
    required_fields = {
        'Initial Capital': 'initial_capital',
        'Floor': 'floor',
        'Risky Ticker': 'risky_ticker',
        'Start Date': 'start_date',
        'End Date': 'end_date',
        'Risk Free Rate': 'risk_free_rate',
        'Pricing Method': 'pricing_method',  
    }
    
    for field, name in required_fields.items():
        if not form_data.get(name):
            return f"{field} is required"

    return None

def comparison_save_form_data_obpi(form_data):
    session['initial_capital'] = float(form_data.get('initial_capital', 10000))
    session['floor'] = float(form_data.get('floor', 0.9))
    session['risky_ticker'] = form_data.get('risky_ticker')
    session['start_date'] = form_data.get('start_date')
    session['end_date'] = form_data.get('end_date')
    session['risk_free_rate'] = float(form_data.get('risk_free_rate', 0.02))
    session['pricing_method'] = form_data.get('pricing_method')

# ---------------------------
# Route for OBPI page
# ---------------------------
@app.route('/obpi', methods=['GET', 'POST'])
def run_obpi():
    if request.method == 'POST':
        form_data = request.form
        error = validate_form_data_obpi(form_data)
        if error:
            flash(error, 'error')
            return redirect(url_for('run_obpi'))
        
        try:
            save_form_data_obpi(form_data)
            pi = PortfolioInsurance(
                initial_capital=float(session['initial_capital']),
                floor=float(session['floor']),
                risky_ticker=session['risky_ticker'],
                start_date=session['start_date'],
                end_date=session['end_date']
            )
            
            # Determine the chosen strategy and pricing method
            if session['strategy'] == 'call+bond':
                results = pi.run_obpi_bond_call(
                    risk_free_rate=float(session['risk_free_rate']),
                    merton='merton jump' if session['pricing_method'] == 'Merton Jump' else 'black-scholes'
                )
                strategy_type = 'call'
            elif session['strategy'] == 'put+underlying asset':
                results = pi.run_obpi_underlying_put(
                    risk_free_rate=float(session['risk_free_rate']),
                    merton='merton jump' if session['pricing_method'] == 'Merton Jump' else 'black-scholes'
                )
                strategy_type = 'put'
            else:
                flash("Invalid strategy selected", 'error')
                return redirect(url_for('run_obpi'))
            
            base_path = os.path.join(os.getcwd(), 'static', 'plots')
            os.makedirs(base_path, exist_ok=True)
            plot_links = pi._plot_generic(results, strategy_type, "obpi_results", base_path)
            
            session['plot_links'] = plot_links
            session['results'] = {
                'n_shares': results.get('n_shares', 0.00),
                'sharpe_portfolio': float(results.get('sharpe_portfolio', 0.00)),
                'mdd_portfolio': float(results.get('mdd_portfolio', 0.00)),
                'final_portfolio_value': float(results.get('portfolio_values', [0.00])[-1]),
                'sharpe_buy_and_hold': float(results.get('sharpe_buy_and_hold', 0.00)),
                'mdd_buy_and_hold': float(results.get('mdd_buy_and_hold', 0.00)),
                'final_buy_and_hold_value': float(results.get('buy_and_hold_values', [0.00])[-1])
            }
            
            print(results)

            return redirect(url_for('run_obpi'))
        
        except Exception as e:
            logging.error(f"Error in OBPI page: {str(e)}", exc_info=True)
            flash(f"Error processing simulation: {str(e)}", 'error')
            return redirect(url_for('run_obpi'))
    
    context = {
        'initial_capital': session.get('initial_capital', 10000),
        'floor': session.get('floor', 0.9),
        'risky_ticker': session.get('risky_ticker', 'SPY'),
        'start_date': session.get('start_date', '2020-01-01'),
        'end_date': session.get('end_date', '2023-12-31'),
        'risk_free_rate': session.get('risk_free_rate', 0.02),
        'pricing_method': session.get('pricing_method', 'Black-Scholes'),
        'strategy': session.get('strategy', 'call+bond'),
        'plot_links': session.get('plot_links'),
        'results': session.get('results'),
        'current_route': 'run_obpi'
    }
    
    return render_template('obpi_page.html', **context)

# ---------------------------
# Route for OBPI Comparison
# ---------------------------
@app.route('/obpi_comparison', methods=['GET', 'POST'])
def run_obpi_comparison():
    if request.method == 'POST':
        form_data = request.form

        error = comparison_validate_form_data_obpi(form_data)
        if error:
            flash(error, 'error')
            return redirect(url_for('run_obpi_comparison'))

        comparison_save_form_data_obpi(form_data)

        try:
            pi = PortfolioInsurance(
                initial_capital=float(session['initial_capital']),
                floor=float(session['floor']),
                risky_ticker=session['risky_ticker'],
                start_date=session['start_date'],
                end_date=session['end_date']
            )

            results_call_bond = pi.run_obpi_bond_call(
                risk_free_rate=float(session['risk_free_rate']),
                merton='merton jump' if session['pricing_method'] == 'Merton Jump' else 'black-scholes'
            )

            results_put_underlying = pi.run_obpi_underlying_put(
                risk_free_rate=float(session['risk_free_rate']),
                merton='merton jump' if session['pricing_method'] == 'Merton Jump' else 'black-scholes'
            )

            logging.debug(f"Results Call + Bond: {results_call_bond}")
            logging.debug(f"Results Put + Underlying: {results_put_underlying}")

            base_path = os.path.join(os.getcwd(), 'static', 'plots')
            os.makedirs(base_path, exist_ok=True)
            plot_links = pi.plot_strategy_comparison(results_call_bond, results_put_underlying, "obpi_comparison_results", base_path)

            session['plot_links'] = {
                'strategy_comparison': plot_links.get('strategy_comparison', '')  # Save just the file name or URL
            }


            session['results'] = {
                'n_shares_call_bond': results_call_bond.get('n_shares', 0),
                'n_shares_put_underlying': results_put_underlying.get('n_shares', 0),
                'sharpe_call_bond': float(results_call_bond.get('sharpe_portfolio', 0.00)),
                'mdd_call_bond': float(results_call_bond.get('mdd_portfolio', 0.00)),
                'call_bond_value': float(results_call_bond.get('portfolio_values', [0.00])[-1]),
                'sharpe_put_underlying': float(results_put_underlying.get('sharpe_portfolio', 0.00)),
                'mdd_put_underlying': float(results_put_underlying.get('mdd_portfolio', 0.00)),
                'put_underlying_value': float(results_put_underlying.get('portfolio_values', [0.00])[-1]),
                'sharpe_buy_and_hold': float(results_call_bond.get('sharpe_buy_and_hold', 0.00)),
                'mdd_buy_and_hold': float(results_call_bond.get('mdd_buy_and_hold', 0.00)),
                'buy_and_hold_value': float(results_call_bond.get('buy_and_hold_values', [0.00])[-1])
            }


            print(results_call_bond['buy_and_hold_values'][-1])


        except Exception as e:
            logging.error(f"Error in OBPI comparison: {str(e)}", exc_info=True)
            flash(f"Error processing simulation: {str(e)}", 'error')
        
        return redirect(url_for('run_obpi_comparison'))

    print(session)
    context = {
        'initial_capital': session.get('initial_capital', 10000),
        'floor': session.get('floor', 0.9),
        'risky_ticker': session.get('risky_ticker', 'SPY'),
        'start_date': session.get('start_date', '2020-01-01'),
        'end_date': session.get('end_date', '2023-12-31'),
        'risk_free_rate': session.get('risk_free_rate', 0.02),
        'pricing_method': session.get('pricing_method', 'Black-Scholes'),
        'plot_links': session.get('plot_links'),
        'results': session.get('results'),
        'current_route': 'run_obpi_comparison'
    }
    return render_template('obpi_comparison.html', **context)



# ---------------------------
# CPPI form validation and data saving functions
# ---------------------------



def validate_form_data_cppi(form_data):
    required_fields = {
        'Initial Capital': 'initial_capital',
        'Floor': 'floor',
        'Multiplier': 'multiplier',
        'Risky Ticker': 'risky_ticker',
        'Start Date': 'start_date',
        'End Date': 'end_date',
        'Rebalance Frequency': 'rebalance_frequency',
        'Risk Free Rate': 'risk_free_rate',
    }
    
    for field, name in required_fields.items():
        if not form_data.get(name):
            return f"{field} is required"

    try:
        initial_capital = float(form_data.get('initial_capital'))
        floor = float(form_data.get('floor'))
        risk_free_rate = float(form_data.get('risk_free_rate'))
        if floor < 0 or floor > 1:
            return "Floor must be between 0 and 1"
        if initial_capital <= 0:
            return "Initial Capital must be positive"
        if risk_free_rate < 0:
            return "Risk Free Rate must be non-negative"
    except ValueError:
        return "All numeric fields must contain valid numbers"
    
    return None

def save_form_data_cppi(form_data):
    session['initial_capital'] = float(form_data.get('initial_capital', 10000))
    session['floor'] = float(form_data.get('floor', 0.9))
    session['multiplier'] = float(form_data.get('multiplier', 3))
    session['risky_ticker'] = form_data.get('risky_ticker')
    session['start_date'] = form_data.get('start_date')
    session['end_date'] = form_data.get('end_date')
    session['rebalance_frequency'] = form_data.get('rebalance_frequency', 1)
    session['risk_free_rate'] = float(form_data.get('risk_free_rate', 0.02))
    



# ---------------------------
# Route for CPPI page (placeholder)
# ---------------------------
@app.route('/cppi', methods=['GET', 'POST'])
def run_cppi():
    if request.method == 'POST':
        form_data = request.form
        error = validate_form_data_cppi(form_data)
        if error:
            flash(error, 'error')
            return redirect(url_for('run_cppi'))
        
        try:
            save_form_data_cppi(form_data)
            pi = PortfolioInsurance(
                initial_capital=float(session['initial_capital']),
                floor=float(session['floor']),
                risky_ticker=session['risky_ticker'],
                start_date=session['start_date'],
                end_date=session['end_date']
            )
            
            results = pi.run_cppi(
                multiplier=float(session['multiplier']),
                rebalance_frequency=int(session['rebalance_frequency']),
                risk_free_rate=float(session['risk_free_rate'])
            )
            
            base_path = os.path.join(os.getcwd(), 'static', 'plots')
            os.makedirs(base_path, exist_ok=True)
            plot_links = pi.plot_cppi_strategy(results, "cppi_results", base_path)
            
            session['plot_links'] = plot_links
            session['results'] = {
                'sharpe_portfolio': results.get('sharpe_portfolio', [0.0]).tolist(),
                'mdd_portfolio': results.get('mdd_portfolio', [0.0]).tolist(),
                'final_portfolio_value': float(results.get('portfolio_values', [0.00])[-1]),
                'sharpe_buy_and_hold': float(results.get('sharpe_buy_and_hold', 0.00)),
                'mdd_buy_and_hold': float(results.get('mdd_buy_and_hold', 0.00)),
                'final_buy_and_hold_value': float(results.get('buy_and_hold_values', [0.00])[-1])
            }
            

            return redirect(url_for('run_cppi'))
        
        except Exception as e:
            logging.error(f"Error in CPPI page: {str(e)}", exc_info=True)
            flash(f"Error processing simulation: {str(e)}", 'error')
            return redirect(url_for('run_cppi'))
        
    
    
    context = {
        'initial_capital': session.get('initial_capital', 10000),
        'floor': session.get('floor', 0.9),
        'multiplier': session.get('multiplier', 3),
        'risky_ticker': session.get('risky_ticker', 'SPY'),
        'start_date': session.get('start_date', '2020-01-01'),
        'end_date': session.get('end_date', '2023-12-31'),
        'rebalance_frequency': session.get('rebalance_frequency', 1),
        'risk_free_rate': session.get('risk_free_rate', 0.02),
        'plot_links': session.get('plot_links'),
        'results': session.get('results'),
        'current_route': 'run_cppi'
    }
    
    return render_template('cppi_page.html', **context)



# ---------------------------
# Comparison form validation and data saving functions
# ---------------------------




def validate_form_data_comparison(form_data):
    required_fields = {
        'Initial Capital': 'initial_capital',
        'Floor': 'floor',
        'Multiplier': 'multiplier',
        'Risky Ticker': 'risky_ticker',
        'Start Date': 'start_date',
        'End Date': 'end_date',
        'Rebalance Frequency': 'rebalance_frequency',
        'Risk Free Rate': 'risk_free_rate',
    }
    
    for field, name in required_fields.items():
        if not form_data.get(name):
            return f"{field} is required"

    try:
        initial_capital = float(form_data.get('initial_capital'))
        floor = float(form_data.get('floor'))
        risk_free_rate = float(form_data.get('risk_free_rate'))
        multiplier = float(form_data.get('multiplier'))
        rebalance_frequency = int(form_data.get('rebalance_frequency', 1))
        if floor < 0 or floor > 1:
            return "Floor must be between 0 and 1"
        if initial_capital <= 0:
            return "Initial Capital must be positive"
        if risk_free_rate < 0:
            return "Risk Free Rate must be non-negative"
        if multiplier <= 0:
            return "Multiplier must be positive"
        if rebalance_frequency <= 0:
            return "Rebalance Frequency must be positive"
    except ValueError:
        return "All numeric fields must contain valid numbers"
    
    return None

def save_form_data_comparison(form_data):
    session['initial_capital'] = float(form_data.get('initial_capital', 10000))
    session['floor'] = float(form_data.get('floor', 0.9))
    session['multiplier'] = float(form_data.get('multiplier', 3))
    session['risky_ticker'] = form_data.get('risky_ticker')
    session['start_date'] = form_data.get('start_date')
    session['end_date'] = form_data.get('end_date')
    session['rebalance_frequency'] = int(form_data.get('rebalance_frequency', 1))
    session['risk_free_rate'] = float(form_data.get('risk_free_rate', 0.02))


# ---------------------------
# Route for Comparison page (placeholder)
# ---------------------------
@app.route('/comparison', methods=['GET', 'POST'])
def run_comparison():
    if request.method == 'POST':
        form_data = request.form
        error = validate_form_data_comparison(form_data)
        if error:
            flash(error, 'error')
            return redirect(url_for('run_comparison'))

        try:
            save_form_data_comparison(form_data)

            pi = PortfolioInsurance(
                initial_capital=session['initial_capital'],
                floor=session['floor'],
                risky_ticker=session['risky_ticker'],
                start_date=session['start_date'],
                end_date=session['end_date']
            )

            base_path = os.path.join(os.getcwd(), 'static', 'plots')
            os.makedirs(base_path, exist_ok=True)

            results = pi.compare_all_strategies(
                risk_free_rate=session['risk_free_rate'],
                cppi_multiplier=session['multiplier'],
                rebalance_frequency=session['rebalance_frequency'],
                base_path=base_path
            )

            session['plot_links'] = results.get('plot_filenames')
            print(session['plot_links'])
            df = results.get('performance_metrics')

            # Convert any NumPy arrays to Python lists to ensure JSON serializability
            session['results'] = {
                'strategies': df['Strategy'].tolist(),
                'sharpe_portfolio': [float(x) if x is not None else None for x in df['Sharpe Ratio'].tolist()],
                'mdd_portfolio': [float(x) if x is not None else None for x in df['Max Drawdown (%)'].tolist()],
                'final_portfolio_value': [float(x) if x is not None else None for x in df['Final Value'].tolist()],
                'total_return': [float(x) if x is not None else None for x in df['Total Return (%)'].tolist()]
            }

            return redirect(url_for('run_comparison'))

        except Exception as e:
            logging.error(f"Error in Comparison page: {str(e)}", exc_info=True)
            flash(f"Error processing simulation: {str(e)}", 'error')
            return redirect(url_for('run_comparison'))

    context = {
        'initial_capital': session.get('initial_capital', 10000),
        'floor': session.get('floor', 0.9),
        'multiplier': session.get('multiplier', 3),
        'risky_ticker': session.get('risky_ticker', 'SPY'),
        'start_date': session.get('start_date', '2020-01-01'),
        'end_date': session.get('end_date', '2023-12-31'),
        'rebalance_frequency': session.get('rebalance_frequency', 1),
        'risk_free_rate': session.get('risk_free_rate', 0.02),
        'plot_links': session.get('plot_links'),
        'results': session.get('results'),
        'current_route': 'run_comparison'
    }

    return render_template('comparison_page.html', **context)


@app.route("/clear_session")
def clear_session():
    session.clear()
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=False)

