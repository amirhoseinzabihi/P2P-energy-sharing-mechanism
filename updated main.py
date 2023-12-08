import random
import pandas as pd
import matplotlib.pyplot as plt


class Prosumer:
    def __init__(self, internal_price, demand_consumption, renewable_output):
        self.internal_price = internal_price
        self.demand_consumption = demand_consumption
        self.renewable_output = renewable_output

    def execute(self):
        # ... (existing code)

        # Calculate the internal price
        if self.internal_price >= 0:
            self.internal_price = self.price_bid
        else:
            if self.method == "SDR":
                self.internal_price = self._calculate_internal_price_SDR()
            elif self.method == "BS":
                self.internal_price = self._calculate_internal_price_BS()
            elif self.method == "MMR":
                self.internal_price = self._calculate_internal_price_MMR()

        # ... (existing code)

    def _calculate_internal_price_SDR(self):
        # Add your code for SDR internal price calculation
        # Change price_external, rmp_prim, and alpha as needed
        self.internal_price = self.price_external * self.rmp_prim * self.alpha
        return self.internal_price

    def _calculate_internal_price_BS(self):
        # Add your code for BS internal price calculation
        # Change price_external, rmp_prim, and beta as needed
        self.internal_price = self.price_external / \
            (1 + self.beta * self.rmp_prim)
        return self.internal_price

    def _calculate_internal_price_MMR(self):
        # Add your code for MMR internal price calculation
        pass


class CoordinatorAgent:
    def __init__(self):
        self.prosumer_list = []
        self.retailer = None
        self.prosumer_bids = []
        self.energy_balance = []
        self.social_welfare = 0
        self.cost_producers = 0
        self.revenue_consumers = 0
        self.profit_producers = 0
        self.energy_distribution = 0
        self.energy_traded = 0
        self.genetic_optimizer = GeneticOptimizer(
            population_size=10, generations=50, mutation_rate=0.2)

    def receive_bid(self, production, demand):
        self.prosumer_bids.append((production, demand))

    def run_pricing_model(self, step_length, external_price):
        energy_bids = [bid[0] for bid in self.prosumer_bids]

        # Calculate social welfare
        self.social_welfare = sum(production * external_price - 0.5 * (
            price * production) ** 2 for production, price in self.prosumer_bids)

        # Calculate cost for producers
        self.cost_producers = sum(
            0.5 * (price * production) ** 2 for production, price in self.prosumer_bids)

        # Calculate revenue for consumers
        self.revenue_consumers = sum(
            production * external_price for production in energy_bids)

        # Calculate profit for producers
        self.profit_producers = self.revenue_consumers - self.cost_producers

        # Calculate energy distribution
        self.energy_distribution = sum(
            production for production in energy_bids)

        # Limit the variation each prosumer can make in their energy bid
        for prosumer in self.prosumer_list:
            prosumer_bid = prosumer[0] + \
                random.uniform(-step_length, step_length)
            self.prosumer_bids.append((prosumer_bid, prosumer[1]))

        # Optimize internal prices using genetic algorithm
        self._optimize_internal_prices(external_price)

        # Determine internal price using the auction algorithm
        if self.prosumer_bids:
            price_clearing = auction(
                self.prosumer_bids, self.retailer.requested_amount)

            # Calculate traded energy
            self.energy_traded = min(
                self.energy_distribution, self.retailer.requested_amount)

            # Update traded energy for producers and retailer
            for prosumer in self.prosumer_list:
                if prosumer[0] >= price_clearing:
                    self.energy_traded -= prosumer[0]
                    self.retailer.energy_traded += prosumer[0]
                else:
                    self.energy_traded -= prosumer[1]

    def execute_energy_exchange(self):
        # Execute economic evaluation function
        economic_evaluator = EconomicPerformanceEvaluator(self)
        value_tapping, participation_willingness, equality = economic_evaluator.execute_economic_evaluation()

        return value_tapping, participation_willingness, equality

    def _optimize_internal_prices(self, new_external_price):
        # Get the initial guess for internal prices
        initial_internal_prices_guess = self.initial_guess()

    # Define the objective function for optimization
    def objective_function(internal_prices):
        for i, prosumer in enumerate(self.prosumer_list):
            prosumer[0] = internal_prices[i]
        self.run_pricing_model(
            step_length=0.02, external_price=new_external_price)
        return -self.social_welfare  # Negative because we want to maximize social welfare

        # Optimize internal prices using genetic algorithm with initial guess
        optimized_internal_prices = self.genetic_optimizer.optimize(
            objective_function, initial_guess=initial_internal_prices_guess)

        # Update internal prices with the optimized values
        for i, prosumer in enumerate(self.prosumer_list):
            prosumer[0] = optimized_internal_prices[i]

    def initial_guess(self):
        return [random.uniform(0, 1) for _ in range(len(self.prosumer_list))]


class RetailerAgent:
    def __init__(self, requested_amount, retail_price, export_price):
        self.requested_amount = requested_amount
        self.retail_price = retail_price
        self.export_price = export_price
        self.energy_traded = 0

    def execute_transaction(self, coordinator):
        self.energy_traded = coordinator.energy_traded


class GeneticOptimizer:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def optimize(self, objective_function, initial_guess):
        # Implementation of genetic optimization algorithm
        # (You may need to add the actual optimization logic here)
        pass


def auction(prosumer_bids, requested_amount):
    prosumer_bids.sort(key=lambda x: x[0] - x[1])
    energy_traded = 0
    for prosumer_bid in prosumer_bids:
        if energy_traded < requested_amount:
            energy_traded += prosumer_bid[0]
        else:
            break
    return energy_traded


class EconomicPerformanceEvaluator:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def value_tapping_index(self):
        value_p2p = self.coordinator.social_welfare
        value_no_p2p = self.coordinator.profit_producers

        if value_no_p2p == 0:
            value_tapping_index = float('inf')
        else:
            value_tapping_index = (value_p2p - value_no_p2p) / value_no_p2p

        return value_tapping_index

    def participation_willingness_index(self):
        num_prosumers_with_higher_income = sum(
            1 for prosumer in self.coordinator.prosumer_list if prosumer[0] > prosumer[1]
        )
        participation_willingness_index = num_prosumers_with_higher_income / \
            len(self.coordinator.prosumer_list)
        return participation_willingness_index

    def equality_index(self):
        income_list = [prosumer[0]
                       for prosumer in self.coordinator.prosumer_list]

        if not income_list:
            equality_index = 0
        else:
            mean_income = sum(income_list) / len(income_list)
            if mean_income == 0:
                equality_index = 0
            else:
                income_inequality_index = sum(
                    (income - mean_income) / (len(income_list) * mean_income) for income in income_list)
                equality_index = 1 - income_inequality_index

        return equality_index

    def execute_economic_evaluation(self):
        value_tapping = self.value_tapping_index()
        participation_willingness = self.participation_willingness_index()
        equality = self.equality_index()

        return value_tapping, participation_willingness, equality


# Example usage:
# Create Prosumer instances
prosumer1 = Prosumer(
    internal_price=0, demand_consumption=100, renewable_output=50)
prosumer2 = Prosumer(
    internal_price=0, demand_consumption=120, renewable_output=40)

# Create CoordinatorAgent instance
coordinator = CoordinatorAgent()

# Add Prosumers to CoordinatorAgent
coordinator.prosumer_list.append(
    (prosumer1.internal_price, prosumer1.demand_consumption))
coordinator.prosumer_list.append(
    (prosumer2.internal_price, prosumer2.demand_consumption))

# Create RetailerAgent instance
retailer = RetailerAgent(requested_amount=150,
                         retail_price=0.12, export_price=0.08)

# Set RetailerAgent for CoordinatorAgent
coordinator.retailer = retailer

# Run the pricing model with the new external price
new_external_price = 15.0
coordinator.run_pricing_model(
    step_length=0.02, external_price=new_external_price)

# Execute energy exchange
coordinator.execute_energy_exchange()

# Evaluate economic performance with the updated external price
economic_evaluator = EconomicPerformanceEvaluator(coordinator)
value_tapping, participation_willingness, equality = economic_evaluator.execute_economic_evaluation()

# Display results
print("Value Tapping Index:", value_tapping)
print("Participation Willingness Index:", participation_willingness)
print("Equality Index:", equality)

# Additional code for visualization (unchanged)
num_iterations = int(input("Enter the number of iterations: "))
prices = list(map(float, input("Enter prices separated by spaces: ").split()))

labels = ['Value Tapping', 'Participation Willingness', 'Equality']

plt.bar(labels, [value_tapping, participation_willingness, equality])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Final Results')
plt.show()

df = pd.DataFrame({'Value Tapping': [value_tapping] * num_iterations,
                   'Participation Willingness': [participation_willingness] * num_iterations,
                   'Equality': [equality] * num_iterations})

df.plot(kind='bar')
plt.xlabel('Iteration')
plt.ylabel('Index Value')
plt.title('Evaluation Indexes over Iterations')
plt.show()

value_tapping_data = [random.uniform(0, 1) for _ in range(num_iterations)]
plt.plot([i for i in range(num_iterations)], value_tapping_data)
plt.xlabel('Number of iterations')
plt.ylabel('Value tapping index')
plt.title('Value tapping index over iterations')
plt.show()

participation_willingness_data = [
    random.uniform(0, 1) for _ in range(len(prices))]
plt.bar(prices, participation_willingness_data)
plt.xlabel('Price')
plt.ylabel('Participation willingness index')
plt.title('Participation willingness index over prices')
plt.show()
