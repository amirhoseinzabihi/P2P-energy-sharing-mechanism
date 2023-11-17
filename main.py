import random
import pandas as pd
import matplotlib.pyplot as plt


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

        # Determine internal price using the auction algorithm
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


class RetailerAgent:
    def __init__(self, requested_amount, retail_price, export_price):
        self.requested_amount = requested_amount
        self.retail_price = retail_price
        self.export_price = export_price
        self.energy_traded = 0

    def execute_transaction(self, coordinator):
        self.energy_traded = coordinator.energy_traded


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
            # Handle the case where value_no_p2p is zero
            # or set to another appropriate value
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
        mean_income = sum(income_list) / len(income_list)

        if len(income_list) == 0:
            # Handle the case where income_list is empty to avoid division by zero
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

        print('Value Tapping Index:', value_tapping)
        print('Participation Willingness Index:', participation_willingness)
        print('Equality Index:', equality)

        return value_tapping, participation_willingness, equality


# Create prosumers
prosumer1 = (0, 10)
prosumer2 = (0, 15)
prosumer3 = (28, 12)
prosumer4 = (35, 18)
prosumer5 = (32, 20)

# Create coordinator agent and add prosumers
coordinator_agent = CoordinatorAgent()
coordinator_agent.prosumer_list.extend(
    [prosumer1, prosumer2, prosumer3, prosumer4, prosumer5])

# Create retailer agent
retailer_agent = RetailerAgent(
    requested_amount=70, retail_price=0.1, export_price=0.05)
coordinator_agent.retailer = retailer_agent
# Run simulation
num_iterations = 100
value_tapping_data = []
participation_willingness_data = []
equality_data = []

external_price = float(input("Please enter the external price: "))
for i in range(num_iterations):
    # Add variability to bids
    for prosumer in coordinator_agent.prosumer_list:
        prosumer_bid = prosumer[0] + random.normalvariate(0, 5)
        coordinator_agent.receive_bid(prosumer_bid, prosumer[1])

    # Run energy exchange
    coordinator_agent.run_pricing_model(
        step_length=0.1, external_price=external_price)

    # Store results
    value_tapping, participation_willingness, equality = coordinator_agent.execute_energy_exchange()
    value_tapping_data.append(value_tapping)
    participation_willingness_data.append(participation_willingness)
    equality_data.append(equality)

# Create a DataFrame
df = pd.DataFrame({
    'Value Tapping': value_tapping_data,
    'Participation Willingness': participation_willingness_data,
    'Equality': equality_data
})

# Plot results using line plot
df.plot(kind='line')
plt.xlabel('Iteration')
plt.ylabel('Index Value')
plt.title('Evaluation Indexes over Iterations')
plt.show()
print(df)
print(df)

# Getting the final values of the desired variables
final_value_tapping = value_tapping_data[-1]
final_participation_willingness = participation_willingness_data[-1]
final_equality = equality_data[-1]

# Putting the values in a list to display in a diagram
results = [final_value_tapping,
           final_participation_willingness, final_equality]

# Required labels for the x-axis
labels = ['Value Tapping', 'Participation Willingness', 'Equality']

# Plot diagram
plt.bar(labels, results)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Final Results')

plt.show()

# Plot results using bar plot
df.plot(kind='bar')
plt.xlabel('Iteration')
plt.ylabel('Index Value')
plt.title('Evaluation Indexes over Iterations')
plt.show()
# Import the matplotlib library

# Plot the graph
plt.plot([i for i in range(num_iterations)], value_tapping_data)
plt.xlabel('Number of iterations')
plt.ylabel('Value tapping index')
plt.title('Value tapping index over iterations')
plt.show()


