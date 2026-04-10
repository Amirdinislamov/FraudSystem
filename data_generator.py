import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import random

NUM_CLIENTS = 600
DAYS = 30
BASE_LAT, BASE_LON = 41.31, 69.24

MCC_STATS = {
    "5411": {"mu": 3.0, "sigma": 0.5},
    "5814": {"mu": 2.0, "sigma": 0.4},
    "5732": {"mu": 6.5, "sigma": 1.2},
    "4511": {"mu": 5.5, "sigma": 0.8},
}

PURPOSES = ["ME2ME", "FAMILY", "PURCHASE", "DEBT_PAYOFF", "INVESTMENT", "CHARITY"]

class Client:
    def __init__(self, cid):
        self.id = cid
        self.profile = np.random.choice(["HOMEBODY", "NIGHT_OWL", "TRAVELER"], p=[0.6, 0.3, 0.1])
        
        self.home_lat = BASE_LAT + np.random.normal(0, 0.05)
        self.home_lon = BASE_LON + np.random.normal(0, 0.05)
        
        if self.profile == "NIGHT_OWL":
            self.mean_hour = np.random.normal(22, 3) % 24
            self.tx_prob = 0.4
        elif self.profile == "TRAVELER":
            self.mean_hour = np.random.normal(14, 6) % 24
            self.tx_prob = 0.5
        else: 
            self.mean_hour = np.random.normal(12, 3) % 24
            self.tx_prob = 0.3

        self.is_compromised = False
        self.compromise_day = -1
        self.is_dropper = False

def generate_stochastic_data():
    clients = [Client(f"USER_{i:03d}") for i in range(NUM_CLIENTS)]
    transactions = []
    start_date = datetime.now() - timedelta(days=DAYS)

    droppers = random.sample(clients, 5)
    for d in droppers: d.is_dropper = True
    
    ato_victims = random.sample([c for c in clients if not c.is_dropper], 15)
    for v in ato_victims:
        v.is_compromised = True
        v.compromise_day = random.randint(10, 25) 

    print(f"Генерация {DAYS} дней органической жизни для {NUM_CLIENTS} клиентов...")

    for day in range(DAYS):
        current_date = start_date + timedelta(days=day)
        
        for c in clients:
            if c.is_compromised and day >= c.compromise_day:
                distance_shift = random.choice([random.uniform(0.1, 0.5), random.uniform(5, 15)])
                hacker_lat = c.home_lat + distance_shift
                hacker_lon = c.home_lon + distance_shift
                
                for _ in range(random.randint(2, 8)):
                    tx_time = current_date.replace(hour=random.randint(0,23), minute=random.randint(0,59))
                    is_p2p = random.random() > 0.5

                    if random.random() < 0.2: 
                        amount = round(np.random.lognormal(1.0, 0.5), 2)
                    else:
                        amount = round(np.random.lognormal(6.0, 1.0), 2)

                    fraud_purpose = np.random.choice(PURPOSES, p=[0.1, 0.1, 0.1, 0.1, 0.5, 0.1]) if is_p2p else None
                    
                    transactions.append({
                        "transaction_id": str(uuid.uuid4()),
                        "client_id": c.id,
                        "amount_usd": amount, 
                        "mcc_code": random.choice(["5732", "4511", "5411"]) if not is_p2p else None,
                        "transfer_purpose": fraud_purpose,
                        "receiver_id": None,
                        "terminal_lat": hacker_lat,
                        "terminal_lon": hacker_lon,
                        "timestamp": tx_time,
                        "is_fraud": 1,
                        "scenario": "Account Takeover (ATO)"
                    })
                continue 

            if random.random() < c.tx_prob:
                for _ in range(random.randint(1, 3)):
                    hour = int(np.random.normal(c.mean_hour, 2)) % 24
                    tx_time = current_date.replace(hour=hour, minute=random.randint(0, 59))
                    
                    if random.random() > 0.98: 
                        lat = c.home_lat + random.uniform(2, 5)
                        lon = c.home_lon + random.uniform(2, 5)
                    else:
                        if c.profile == "TRAVELER" and random.random() > 0.8:
                            c.home_lat += random.uniform(-1, 1)
                        lat = c.home_lat + np.random.normal(0, 0.02)
                        lon = c.home_lon + np.random.normal(0, 0.02)

                    is_p2p = random.random() < 0.2
                    amount = 0

                    if is_p2p:
                        purpose = np.random.choice(PURPOSES, p=[0.3, 0.2, 0.2, 0.2, 0.08, 0.02])
                        amount = round(np.random.lognormal(4.0, 1.0), 2)
                        
                        if random.random() < 0.02: 
                            receiver = random.choice(droppers).id
                            is_fraud = 1
                            scenario = "Organic Dropper Network"
                        else:
                            receiver = random.choice(clients).id
                            is_fraud = 0
                            scenario = "Normal"
                    else:
                        mcc = random.choice(list(MCC_STATS.keys()))
                        amount = round(np.random.lognormal(MCC_STATS[mcc]["mu"], MCC_STATS[mcc]["sigma"]), 2)
                        purpose, receiver = None, None
                        
                        if random.random() < 0.01:
                            amount = amount * random.uniform(5, 10) 
                            
                        is_fraud, scenario = 0, "Normal"

                    transactions.append({
                        "transaction_id": str(uuid.uuid4()),
                        "client_id": c.id,
                        "amount_usd": max(1.0, amount),
                        "mcc_code": mcc if not is_p2p else None,
                        "transfer_purpose": purpose,
                        "receiver_id": receiver,
                        "terminal_lat": lat,
                        "terminal_lon": lon,
                        "timestamp": tx_time,
                        "is_fraud": is_fraud,
                        "scenario": scenario
                    })

    df = pd.DataFrame(transactions).sort_values("timestamp")
    df.to_csv("heavy_fraud_data.csv", index=False)
    
    print(f"✅ Готово! Сгенерировано {len(df)} транзакций.")
    print("Распределение фрода:")
    print(df[df['is_fraud'] == 1]['scenario'].value_counts())

if __name__ == "__main__":
    generate_stochastic_data()