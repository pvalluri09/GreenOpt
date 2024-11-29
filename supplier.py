import numpy as np
import streamlit as st

# Define the supplier class to store supplier-related information
class Supplier:
    def __init__(self, name, price, capacity, weight):
        self.name = name
        self.price = price
        self.capacity = capacity
        self.weight = weight

# Function to create the AHP pairwise comparison matrix and calculate the weights
def calculate_ahp_weights():
    price_vs_delivery = 3
    price_vs_environment = 5
    price_vs_quality = 7
    delivery_vs_environment = 3
    delivery_vs_quality = 5
    environment_vs_quality = 1 / 3

    pairwise_matrix = np.array([
        [1, price_vs_delivery, price_vs_environment, price_vs_quality],
        [1 / price_vs_delivery, 1, delivery_vs_environment, delivery_vs_quality],
        [1 / price_vs_environment, 1 / delivery_vs_environment, 1, environment_vs_quality],
        [1 / price_vs_quality, 1 / delivery_vs_quality, 1 / environment_vs_quality, 1]
    ])

    column_sums = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / column_sums

    criteria_weights = normalized_matrix.mean(axis=1)

    return criteria_weights

# Function to calculate weighted rating for each supplier based on AHP weights and user ratings
def calculate_weighted_supplier_scores(ahp_weights, supplier_ratings):
    supplier_weights = []
    criteria = ['Price', 'Quality', 'Delivery', 'Environment']

    for i in range(1, 4):  # For 3 suppliers
        total_weight = 0
        for j, criterion in enumerate(criteria):
            total_weight += ahp_weights[j] * supplier_ratings[f'Supplier {i}'][criterion]
        supplier_weights.append(total_weight)

    return supplier_weights

# Function to calculate Total Value of Purchase (TVP) based on AHP weights
def calculate_tvp(order_quantities, suppliers):
    penalty_factor = 5
    tvp = sum(sup.weight * order_quantities[i] for i, sup in enumerate(suppliers))
    return tvp * penalty_factor

# Function to calculate Total Profit of Purchase (TPP)
def calculate_tpp(order_quantities, suppliers, demand, inventory, shortage):
    revenue_per_item = 30
    holding_cost = 1
    shortage_cost = 12

    revenue = revenue_per_item * min(sum(order_quantities), demand)
    holding = holding_cost * inventory
    shortage_costs = shortage_cost * shortage
    purchasing_cost = sum(sup.price[i] * order_quantities[i] for i, sup in enumerate(suppliers))
    tpp = revenue - holding - shortage_costs - purchasing_cost
    return tpp

# Function to print the optimal order and value for each period
def get_optimal_orders_and_values(demand_scenarios, suppliers, periods):
    # Initialize DP table and order tracking
    max_inventory = 10
    max_shortage = 10
    dp_table = [[[[float('-inf') for _ in range(len(demand_scenarios))] for _ in range(max_shortage + 1)] for _ in range(max_inventory + 1)] for _ in range(periods)]
    order_tracking = [[[[None for _ in range(len(demand_scenarios))] for _ in range(max_shortage + 1)] for _ in range(max_inventory + 1)] for _ in range(periods)]

    # Base case initialization
    for scenario in range(len(demand_scenarios)):
        for inventory in range(max_inventory + 1):
            for shortage in range(max_shortage + 1):
                dp_table[periods - 1][inventory][shortage][scenario] = 0  # No value in the last period

    # Populate the DP table
    for period in range(periods - 1, -1, -1):
        for inventory in range(max_inventory + 1):
            for shortage in range(max_shortage + 1):
                for scenario in range(len(demand_scenarios)):
                    demand = demand_scenarios[scenario][period]

                    for order_s1 in range(min(int(suppliers[0].capacity[period]), int(demand - inventory)) + 1):
                        for order_s2 in range(min(int(suppliers[1].capacity[period]), int(demand - inventory - order_s1)) + 1):
                            for order_s3 in range(min(int(suppliers[2].capacity[period]), int(demand - inventory - order_s1 - order_s2)) + 1):
                                order_quantities = [order_s1, order_s2, order_s3]

                                total_order = sum(order_quantities)
                                new_inventory = max(0, int(inventory + total_order - demand))
                                new_shortage = max(0, int(demand - (inventory + total_order)))

                                new_inventory = min(new_inventory, max_inventory)
                                new_shortage = min(new_shortage, max_shortage)

                                tvp = calculate_tvp(order_quantities, suppliers)
                                tpp = calculate_tpp(order_quantities, suppliers, demand, new_inventory, new_shortage)

                                future_value = dp_table[period + 1][new_inventory][new_shortage][scenario] if period + 1 < periods else 0
                                current_value = tvp + tpp + future_value

                                if current_value > dp_table[period][inventory][shortage][scenario]:
                                    dp_table[period][inventory][shortage][scenario] = current_value
                                    order_tracking[period][inventory][shortage][scenario] = order_quantities

    return dp_table, order_tracking

# Streamlit UI
st.title("Supplier Evaluation and Order Optimization")

# Collect Supplier Ratings
st.subheader("Supplier Ratings")
supplier_ratings = {}
for i in range(1, 4):  # For 3 suppliers
    supplier_ratings[f'Supplier {i}'] = {
        'Price': st.slider(f'Supplier {i} - Price', 1, 5, 1),
        'Quality': st.slider(f'Supplier {i} - Quality', 1, 5, 1),
        'Delivery': st.slider(f'Supplier {i} - Delivery', 1, 5, 1),
        'Environment': st.slider(f'Supplier {i} - Environment', 1, 5, 1)
    }

# Collect Demand Scenarios
st.subheader("Demand Scenarios")
demand_scenarios = []
for period in range(3):  # For 3 periods
    demand = st.number_input(f"Enter demand for period {period + 1}", min_value=0, value=0)
    if period == 0 or len(demand_scenarios) == 0:
        demand_scenarios.append([demand])
    else:
        demand_scenarios[0].append(demand)

if st.button("Calculate Optimal Orders"):
    criteria_weights = calculate_ahp_weights()
    supplier_weights = calculate_weighted_supplier_scores(criteria_weights, supplier_ratings)

    suppliers = [
        Supplier("Supplier 1", price=[8, 8, 8], capacity=[3, 3, 3], weight=supplier_weights[0]),
        Supplier("Supplier 2", price=[5, 5, 4], capacity=[3, 3, 3], weight=supplier_weights[1]),
        Supplier("Supplier 3", price=[5, 8, 8], capacity=[5, 5, 5], weight=supplier_weights[2]),
    ]

    periods = len(demand_scenarios[0])
    dp_table, order_tracking = get_optimal_orders_and_values(demand_scenarios, suppliers, periods)

    # Display results
    current_inventory = 0
    current_shortage = 0
    total_value = 0

    st.subheader("Optimal Orders and Values:")
    for period in range(periods):
        optimal_order = order_tracking[period][current_inventory][current_shortage][0]
        optimal_value = dp_table[period][current_inventory][current_shortage][0]

        demand = demand_scenarios[0][period]
        total_order = sum(optimal_order)
        new_inventory = max(0, current_inventory + total_order - demand)
        new_shortage = max(0, demand - (current_inventory + total_order))

        st.write(f"Period {period + 1}:")
        st.write(f"  Optimal Order: {optimal_order}")
        st.write(f"  Value: {optimal_value}")

        current_inventory = new_inventory
        current_shortage = new_shortage
        total_value += optimal_value

    st.write(f"Total Value over all periods: {total_value}")
