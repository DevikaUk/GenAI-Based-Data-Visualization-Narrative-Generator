"""
generate_kaggle_retail.py — Generates a realistic sample of the famous Kaggle Online Retail Dataset
"""
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

products = [
    ("85123A", "White Hanging Heart T-Light Holder", 2.55, "Home Decor"),
    ("71053", "White Metal Lantern", 3.39, "Home Decor"),
    ("84406B", "Cream Cupid Hearts Coat Hanger", 2.75, "Home Decor"),
    ("84029G", "Knitted Union Flag Hot Water Bottle", 3.39, "Seasonal"),
    ("84029E", "Red Woolly Hottie White Heart.", 3.39, "Seasonal"),
    ("22752", "Set 7 Babushka Nesting Boxes", 7.65, "Toys & Games"),
    ("21730", "Glass Star Frosted T-Light Holder", 4.25, "Home Decor"),
    ("22633", "Hand Warmer Union Jack", 1.85, "Accessories"),
    ("22632", "Hand Warmer Red Retrospot", 1.85, "Accessories"),
    ("84879", "Assorted Colour Bird Ornament", 1.69, "Gifts"),
    ("22745", "Poppy's Playhouse Bedroom", 2.10, "Toys & Games"),
    ("22748", "Poppy's Playhouse Kitchen", 2.10, "Toys & Games"),
    ("22749", "Feltcraft Princess Charlotte Doll", 3.75, "Crafts"),
    ("22310", "Ivory Knitting Mug", 1.65, "Kitchenware"),
    ("84969", "Box Of 6 Assorted Colour Teaspoons", 4.25, "Kitchenware"),
    ("20725", "Lunch Bag Red Retrospot", 1.65, "Kitchenware"),
    ("22383", "Lunch Bag Suzi Doodle", 1.65, "Kitchenware"),
    ("22384", "Lunch Bag Pink Polkadot", 1.65, "Kitchenware"),
    ("20727", "Lunch Bag Black Skull.", 1.65, "Kitchenware"),
    ("20728", "Lunch Bag Cars Blue", 1.65, "Kitchenware"),
    ("22492", "Mini Painter's Set 25 Pieces", 0.65, "Crafts"),
    ("22086", "Paper Chain Kit 50's Christmas", 2.55, "Seasonal"),
    ("21212", "Pack Of 72 Retrospot Cake Cases", 0.55, "Bakery"),
    ("22197", "Small Popcorn Holder", 0.85, "Party Supplies"),
    ("21977", "Pack Of 60 Pink Paisley Cake Cases", 0.55, "Bakery"),
    ("84991", "60 Teatime Fairy Cake Cases", 0.55, "Bakery"),
    ("21213", "Pack Of 72 Skull Cake Cases", 0.55, "Bakery"),
    ("22616", "Pack Of 12 London Tissues", 0.29, "Essentials"),
    ("22629", "Spaceboy Mini Back Pack", 4.15, "Bags"),
    ("22630", "Dolly Girl Mini Backpack", 4.15, "Bags"),
]

countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Australia", "Portugal"]
country_weights = [0.65, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]

start_date = datetime(2023, 1, 1)
records = []

for i in range(1, 1501):
    inv_no = f"INV-{536365 + (i // 3):06d}"
    stock_code, desc, base_price, dept = random.choice(products)
    qty = int(np.random.choice([1, 2, 3, 4, 6, 10, 12, 24, 48], p=[0.3, 0.25, 0.15, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01]))
    unit_price = round(base_price * random.uniform(0.9, 1.1), 2)
    inv_date = start_date + timedelta(days=random.randint(0, 364), hours=random.randint(8, 19), minutes=random.randint(0, 59))
    country = random.choices(countries, weights=country_weights)[0]
    cust_id = f"CUST-{random.randint(12000, 18500)}"
    
    records.append({
        "InvoiceNo": inv_no,
        "StockCode": stock_code,
        "Description": desc,
        "Department": dept,
        "Quantity": qty,
        "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M"),
        "UnitPrice": unit_price,
        "CustomerID": cust_id,
        "Country": country
    })

df = pd.DataFrame(records)
df.sort_values("InvoiceDate").to_csv("data/online_retail_kaggle.csv", index=False)
print("Saved data/online_retail_kaggle.csv with shape:", df.shape)
