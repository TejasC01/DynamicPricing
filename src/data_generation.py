import random
import pandas as pd

random.seed(42)


def generate_data(n=1000, save=False, save_path="../data/demand_data.csv"):
    data = []

    for _ in range(n):
        product_id = random.randint(1, 50)
        price = random.uniform(10, 100)
        day_of_week = random.randint(0, 6)
        stock_level = random.randint(20, 100)
        days_to_expiry = random.randint(1, 10)

        base_demand = 50
        price_effect = -0.5 * price
        weekend_boost = 10 if day_of_week in [5, 6] else 0
        expiry_boost = 15 if days_to_expiry < 3 else 0

        units_sold = base_demand + price_effect + weekend_boost + expiry_boost
        units_sold += random.uniform(-5, 5)

        stock_limit = min(stock_level, base_demand)
        units_sold = int(max(0, min(units_sold, stock_limit)))

        data.append([
            product_id,
            price,
            day_of_week,
            stock_level,
            days_to_expiry,
            units_sold
        ])

    df = pd.DataFrame(data, columns=[
        "product_id",
        "price",
        "day_of_week",
        "stock_level",
        "days_to_expiry",
        "units_sold"
    ])

    if save:
        df.to_csv(save_path, index=False)

    return df