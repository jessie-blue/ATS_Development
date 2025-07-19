
import math
from datetime import date, datetime
from scipy.stats import norm


class Option:
    """
    Represents a financial option (call or put).

    Attributes:
        underlying_asset (str): The ticker symbol of the underlying asset (e.g., 'AAPL').
        option_type (str): The type of option ('call' or 'put').  Case-insensitive.
        strike_price (float): The strike price of the option.
        expiration_date (date): The expiration date of the option.
        current_price (float): The current market price of the underlying asset.
        risk_free_rate (float): The risk-free interest rate (annualized, e.g., 0.05 for 5%).
        volatility (float): The volatility of the underlying asset (annualized, e.g., 0.2 for 20%).
    """
    def __init__(self, underlying_asset, option_type, strike_price, expiration_date,
                 current_price, risk_free_rate, volatility):
        self.underlying_asset = underlying_asset
        self.option_type = option_type.lower()  # Ensure lowercase for consistency
        if self.option_type not in ('call', 'put'):
            raise ValueError("Option type must be 'call' or 'put'")
        self.strike_price = strike_price
        self.expiration_date = expiration_date
        self.current_price = current_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def time_to_expiration(self):
        """
        Calculates the time to expiration in years.

        Returns:
            float: Time to expiration in years.
        """
        today = date.today()
        time_diff = (self.expiration_date - today).days
        return time_diff / 365.0

    def display_details(self):
        """
        Prints the details of the option.  Good for command-line output,
        and useful for debugging.  This can be adapted for GUI display.
        """
        print(f"Underlying Asset: {self.underlying_asset}")
        print(f"Option Type: {self.option_type.capitalize()}")
        print(f"Strike Price: ${self.strike_price:.2f}")
        print(f"Expiration Date: {self.expiration_date}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Risk-Free Rate: {self.risk_free_rate:.4f}")
        print(f"Volatility: {self.volatility:.4f}")
        print(f"Time to Expiration: {self.time_to_expiration():.4f} years")

    def calculate_intrinsic_value(self):
        """
        Calculates the intrinsic value of the option.

        Returns:
            float: The intrinsic value.
        """
        if self.option_type == 'call':
            return max(0, self.current_price - self.strike_price)
        else:  # put option
            return max(0, self.strike_price - self.current_price)
        
        

class OptionPricer:
    """
    Provides methods for pricing options using different models.
    This class is designed to be stateless (no instance variables), so it
    can be easily used in a multi-threaded or distributed environment.
    """
    @staticmethod
    def black_scholes(option):
        """
        Calculates the price of an option using the Black-Scholes model.

        Args:
            option (Option): An Option object.

        Returns:
            float: The calculated option price.

        Raises:
            ValueError: If volatility or time to expiration is zero.
        """
        s = option.current_price
        k = option.strike_price
        r = option.risk_free_rate
        t = option.time_to_expiration()
        v = option.volatility

        if v == 0:
            raise ValueError("Volatility cannot be zero")
        if t <= 0:
            return option.calculate_intrinsic_value()

        try:
            d1 = (math.log(s / k) + (r + 0.5 * v ** 2) * t) / (v * math.sqrt(t))
            d2 = d1 - v * math.sqrt(t)
            if option.option_type == 'call':
                price = s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
            else:  # put option
                price = k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
            return price
        except Exception as e:
            print(f"Error in black_scholes: {e}")
            return 0  # Handle errors gracefully

    @staticmethod
    def monte_carlo_simulation(option, num_simulations=10000):
        """
        Estimates the option price using Monte Carlo simulation.

        Args:
            option (Option): An Option object.
            num_simulations (int): The number of simulations to run.  Defaults to 10000.

        Returns:
            float: The estimated option price.
        """
        s = option.current_price
        k = option.strike_price
        r = option.risk_free_rate
        t = option.time_to_expiration()
        v = option.volatility
        option_type = option.option_type

        total_payoff = 0
        for _ in range(num_simulations):
            # Generate a random stock price at expiration
            z = norm.rvs()  # Standard normal random variable
            st = s * math.exp((r - 0.5 * v ** 2) * t + v * math.sqrt(t) * z)

            # Calculate the payoff of the option
            if option_type == 'call':
                payoff = max(0, st - k)
            else:
                payoff = max(0, k - st)
            total_payoff += payoff

        # Discount the average payoff to present value
        option_price = (total_payoff / num_simulations) * math.exp(-r * t)
        return option_price
    
    
    
def main():
    """
    Main function to demonstrate the option pricing tool.  This would be
    replaced by a GUI or backend integration in a full application.
    """
    # Example usage:
    expiration_date = date(2025, 12, 20)
    option = Option(underlying_asset='AAPL',
                    option_type='call',
                    strike_price=190.0,
                    expiration_date=expiration_date,
                    current_price=180.0,
                    risk_free_rate=0.05,
                    volatility=0.20)

    option.display_details()

    pricer = OptionPricer()
    try:
        black_scholes_price = pricer.black_scholes(option)
        print(f"Black-Scholes Price: ${black_scholes_price:.2f}")
    except ValueError as e:
        print(f"Error: {e}")

    monte_carlo_price = pricer.monte_carlo_simulation(option)
    print(f"Monte Carlo Price: ${monte_carlo_price:.2f}")
    option2 = Option(underlying_asset='AAPL',
                    option_type='put',
                    strike_price=190.0,
                    expiration_date=expiration_date,
                    current_price=180.0,
                    risk_free_rate=0.05,
                    volatility=0.20)
    option2.display_details()
    try:
        black_scholes_price2 = pricer.black_scholes(option2)
        print(f"Black-Scholes Price: ${black_scholes_price2:.2f}")
    except ValueError as e:
        print(f"Error: {e}")

    monte_carlo_price2 = pricer.monte_carlo_simulation(option2)
    print(f"Monte Carlo Price: ${monte_carlo_price2:.2f}")

if __name__ == "__main__":
    main()