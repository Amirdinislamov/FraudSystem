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

def generate_random_ip():
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

class Client:
    def __init__(self, cid):
        self.id = cid
        self.profile = np.random.choice(["HOMEBODY", "NIGHT_OWL", "TRAVELER"], p=[0.6, 0.3, 0.1])
        
        self.home_lat = BASE_LAT + np.random.normal(0, 0.05)
        self.home_lon = BASE_LON + np.random.normal(0, 0.05)
        
        # Основные девайсы и IP (для нормального поведения)
        self.devices = [str(uuid.uuid4())[:8] for _ in range(random.randint(1, 2))]
        self.home_ip = generate_random_ip()
        
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

    droppers = random.sample(clients, 10)
    for d in droppers: d.is_dropper = True
    
    ato_victims = random.sample([c for c in clients if not c.is_dropper], 15)
    for v in ato_victims:
        v.is_compromised = True
        v.compromise_day = random.randint(10, 25) 

    # Подготовим бот-ферму: один хакерский девайс и IP для всех скомпрометированных жертв
    bot_farm_device = "HACK_DEV_" + str(uuid.uuid4())[:8]
    bot_farm_ip = generate_random_ip()

    print(f"Генерация {DAYS} дней органической жизни для {NUM_CLIENTS} клиентов...")

    # Для генерации циклов (Smurfing): Отмывание денег
    cycle_rings = []
    for _ in range(5): # 5 колец отмывания
        ring_size = random.randint(3, 5)
        cycle_rings.append(random.sample([c for c in clients if not c.is_dropper], ring_size))

    for day in range(DAYS):
        current_date = start_date + timedelta(days=day)
        
        # 1. Циклы отмывания (Smurfing) раз в несколько дней
        if day % 7 == 0:
            for ring in cycle_rings:
                cycle_amount = round(np.random.uniform(500, 2000), 2)
                for i in range(len(ring)):
                    sender = ring[i]
                    receiver = ring[(i + 1) % len(ring)]
                    tx_time = current_date.replace(hour=12, minute=random.randint(0, 59)) + timedelta(minutes=i*10)
                    
                    transactions.append({
                        "transaction_id": str(uuid.uuid4()),
                        "client_id": sender.id,
                        "amount_usd": cycle_amount, 
                        "mcc_code": None,
                        "transfer_purpose": "ME2ME",
                        "receiver_id": receiver.id,
                        "terminal_lat": sender.home_lat,
                        "terminal_lon": sender.home_lon,
                        "ip_address": sender.home_ip,
                        "device_id": sender.devices[0],
                        "timestamp": tx_time,
                        "is_fraud": 1,
                        "scenario": "Smurfing Cycle"
                    })

        for c in clients:
            # 2. Account Takeover (Bot Farm)
            if c.is_compromised and day >= c.compromise_day:
                distance_shift = random.choice([random.uniform(0.1, 0.5), random.uniform(5, 15)])
                hacker_lat = c.home_lat + distance_shift
                hacker_lon = c.home_lon + distance_shift
                
                for _ in range(random.randint(2, 5)):
                    tx_time = current_date.replace(hour=random.randint(0,23), minute=random.randint(0,59))
                    is_p2p = random.random() > 0.5

                    if random.random() < 0.2: 
                        amount = round(np.random.lognormal(1.0, 0.5), 2)
                    else:
                        amount = round(np.random.lognormal(6.0, 1.0), 2)

                    fraud_purpose = np.random.choice(PURPOSES, p=[0.1, 0.1, 0.1, 0.1, 0.5, 0.1]) if is_p2p else None
                    receiver = random.choice(droppers).id if is_p2p else None

                    transactions.append({
                        "transaction_id": str(uuid.uuid4()),
                        "client_id": c.id,
                        "amount_usd": amount, 
                        "mcc_code": random.choice(["5732", "4511", "5411"]) if not is_p2p else None,
                        "transfer_purpose": fraud_purpose,
                        "receiver_id": receiver,
                        "terminal_lat": hacker_lat,
                        "terminal_lon": hacker_lon,
                        "ip_address": bot_farm_ip,
                        "device_id": bot_farm_device,
                        "timestamp": tx_time,
                        "is_fraud": 1,
                        "scenario": "Bot Farm ATO"
                    })
                continue 

            # 3. Organic Transactions
            if random.random() < c.tx_prob:
                for _ in range(random.randint(1, 3)):
                    hour = int(np.random.normal(c.mean_hour, 2)) % 24
                    tx_time = current_date.replace(hour=hour, minute=random.randint(0, 59))
                    
                    if random.random() > 0.98: 
                        lat = c.home_lat + random.uniform(2, 5)
                        lon = c.home_lon + random.uniform(2, 5)
                    else:
                        lat = c.home_lat + np.random.normal(0, 0.02)
                        lon = c.home_lon + np.random.normal(0, 0.02)

                    is_p2p = random.random() < 0.2
                    amount = 0
                    receiver = None
                    mcc = None
                    purpose = None

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
                        
                        if random.random() < 0.01:
                            amount = amount * random.uniform(5, 10) 
                            
                        is_fraud, scenario = 0, "Normal"

                    transactions.append({
                        "transaction_id": str(uuid.uuid4()),
                        "client_id": c.id,
                        "amount_usd": max(1.0, amount),
                        "mcc_code": mcc,
                        "transfer_purpose": purpose,
                        "receiver_id": receiver,
                        "terminal_lat": lat,
                        "terminal_lon": lon,
                        "ip_address": c.home_ip,
                        "device_id": random.choice(c.devices),
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