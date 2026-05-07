import networkx as nx
from datetime import datetime, timedelta
from typing import Optional, Tuple
from app.schemas import TransactionPayload


class TransactionGraph:
    """
    Гетерогенный граф транзакций.

    Типы узлов:
        - client  : участник транзакции (отправитель / получатель)
        - device  : устройство (device_id)
        - ip      : IP-адрес

    Типы рёбер:
        - transfer      : денежный перевод client → client
        - uses_device   : client → device
        - uses_ip       : client → ip

    Граф используется для детекции:
        1. Дропперов       — узел принимает деньги от ≥3 уникальных отправителей
        2. Смёрфинг-колец  — циклические пути длиной ≤ max_depth
        3. Бот-ферм        — одно устройство / IP разделяют ≥N клиентов
        4. PageRank-хабов  — клиенты с аномально высоким влиянием в сети переводов
        5. Burst-активности — резкий рост исходящих переводов за короткое окно
    """

    WINDOW_DAYS = 7

    def __init__(self):
        self.G = nx.MultiDiGraph()

        self._transfer_graph: Optional[nx.DiGraph] = None
        self._transfer_graph_dirty: bool = False

        self._pagerank_cache: dict = {}
        self._pagerank_counter: int = 0
        self._PAGERANK_RECALC_EVERY = 50  


    def add_transaction(self, tx: TransactionPayload):
        if not self.G.has_node(tx.client_id):
            self.G.add_node(tx.client_id, type="client")

        if tx.receiver_id:
            if not self.G.has_node(tx.receiver_id):
                self.G.add_node(tx.receiver_id, type="client")

            self.G.add_edge(
                tx.client_id,
                tx.receiver_id,
                key=str(tx.transaction_id),
                amount=tx.amount_usd,
                type="transfer",
                timestamp=tx.timestamp,
            )
            self._transfer_graph_dirty = True
            self._pagerank_counter += 1

        if tx.device_id:
            device_node = f"DEV_{tx.device_id}"
            if not self.G.has_node(device_node):
                self.G.add_node(device_node, type="device")
            if not self.G.has_edge(tx.client_id, device_node):
                self.G.add_edge(tx.client_id, device_node, type="uses_device")

        if tx.ip_address:
            ip_node = f"IP_{tx.ip_address}"
            if not self.G.has_node(ip_node):
                self.G.add_node(ip_node, type="ip")
            if not self.G.has_edge(tx.client_id, ip_node):
                self.G.add_edge(tx.client_id, ip_node, type="uses_ip")


    def _get_transfer_graph(self) -> nx.DiGraph:
        """
        Возвращает кешированный подграф только из transfer-рёбер.
        Пересобирается только при изменениях (lazy rebuild).
        """
        if self._transfer_graph is None or self._transfer_graph_dirty:
            tg = nx.DiGraph()
            for u, v, d in self.G.edges(data=True):
                if d.get("type") == "transfer":
                    if tg.has_edge(u, v):
                        tg[u][v]["weight"] += d.get("amount", 1.0)
                        tg[u][v]["count"] += 1
                    else:
                        tg.add_edge(u, v, weight=d.get("amount", 1.0), count=1)
            self._transfer_graph = tg
            self._transfer_graph_dirty = False
        return self._transfer_graph

    def _get_recent_transfer_graph(self, since: datetime) -> nx.DiGraph:
        """Подграф только из «свежих» рёбер (для PageRank по скользящему окну)."""
        tg = nx.DiGraph()
        for u, v, d in self.G.edges(data=True):
            if d.get("type") == "transfer":
                ts = d.get("timestamp")
                if ts and ts >= since:
                    if tg.has_edge(u, v):
                        tg[u][v]["weight"] += d.get("amount", 1.0)
                    else:
                        tg.add_edge(u, v, weight=d.get("amount", 1.0))
        return tg


    def _refresh_pagerank_if_needed(self, current_time: Optional[datetime] = None):
        """Пересчитывает PageRank каждые N транзакций."""
        if self._pagerank_counter >= self._PAGERANK_RECALC_EVERY:
            tg = self._get_transfer_graph()
            if tg.number_of_nodes() > 0:
                try:
                    self._pagerank_cache = nx.pagerank(
                        tg, alpha=0.85, weight="weight", max_iter=100
                    )
                except nx.PowerIterationFailedConvergence:
                    self._pagerank_cache = {}
            self._pagerank_counter = 0
        
    
    def detect_cycle(self, source_id: str, target_id: str, max_depth: int = 5) -> bool:
        """
        Проверяет, образует ли перевод source→target замкнутый цикл.
        Использует BFS с ограничением глубины (cutoff).
        """
        if not source_id or not target_id:
            return False

        tg = self._get_transfer_graph()

        if not (tg.has_node(source_id) and tg.has_node(target_id)):
            return False

        try:
            lengths = nx.single_source_shortest_path_length(
                tg, source=target_id, cutoff=max_depth
            )

            return source_id in lengths

        except nx.NodeNotFound:
            return False
        except Exception:
            return False

    def get_in_degree(self, client_id: str) -> int:
        """
        Количество уникальных отправителей к данному клиенту.
        Используется для детекции дропперов (hub-and-spoke паттерн).
        """
        if not self.G.has_node(client_id):
            return 0
        senders = {
            u for u, v, d in self.G.in_edges(client_id, data=True)
            if d.get("type") == "transfer"
        }
        return len(senders)

    def get_shared_device_count(self, device_id: str) -> int:
        """Количество клиентов, использующих одно устройство."""
        if not device_id:
            return 0
        node = f"DEV_{device_id}"
        if not self.G.has_node(node):
            return 0
        clients = {
            u for u, v, d in self.G.in_edges(node, data=True)
            if d.get("type") == "uses_device"
        }
        return len(clients)

    def get_shared_ip_count(self, ip_address: str) -> int:
        """Количество клиентов, использующих один IP-адрес."""
        if not ip_address:
            return 0
        node = f"IP_{ip_address}"
        if not self.G.has_node(node):
            return 0
        clients = {
            u for u, v, d in self.G.in_edges(node, data=True)
            if d.get("type") == "uses_ip"
        }
        return len(clients)

    def get_pagerank_score(self, client_id: str) -> float:
        """
        PageRank клиента в сети денежных переводов.

        Высокий PageRank означает, что клиент получает деньги от многих
        других «влиятельных» узлов — типичный признак дроппера или
        центрального узла в схеме отмывания.

        Возвращает значение в [0, 1]. Среднее по графу ≈ 1/N.
        """
        self._refresh_pagerank_if_needed()
        return self._pagerank_cache.get(client_id, 0.0)

    def get_out_degree_burst(
        self, client_id: str, current_time: datetime, window_hours: int = 1
    ) -> int:
        """
        Количество ИСХОДЯЩИХ переводов от клиента за последние window_hours часов.

        Резкий рост исходящих переводов за короткое время — признак
        компрометации аккаунта (ATO — Account TakeOver).
        """
        if not self.G.has_node(client_id):
            return 0

        cutoff = current_time - timedelta(hours=window_hours)
        count = 0
        for u, v, d in self.G.out_edges(client_id, data=True):
            if d.get("type") == "transfer":
                ts = d.get("timestamp")
                if ts and ts >= cutoff:
                    count += 1
        return count

    def get_fan_out(self, client_id: str) -> int:
        """
        Количество уникальных получателей у данного клиента.

        Высокий fan-out при небольшой общей истории — признак смёрфинга
        (дробление крупной суммы на множество мелких получателей).
        """
        if not self.G.has_node(client_id):
            return 0
        tg = self._get_transfer_graph()
        if not tg.has_node(client_id):
            return 0
        return tg.out_degree(client_id)

    def get_recent_unique_senders(
        self, client_id: str, current_time: datetime, window_hours: int = 1
    ) -> int:
        """
        Количество уникальных отправителей, которые перевели деньги
        данному получателю за последние window_hours часов.

        Эта метрика отделяет «дропперов в моменте» от просто популярных
        получателей: даже если получатель за всю историю получал переводы
        от десятков людей, опасным он становится тогда, когда поступления
        от множества разных людей идут в течение короткого окна.
        """
        if not self.G.has_node(client_id):
            return 0
        cutoff = current_time - timedelta(hours=window_hours)
        senders = set()
        for u, v, d in self.G.in_edges(client_id, data=True):
            if d.get("type") == "transfer":
                ts = d.get("timestamp")
                if ts and ts >= cutoff:
                    senders.add(u)
        return len(senders)

    def get_total_outflow(self, client_id: str, current_time: datetime, window_hours: int = 24) -> float:
        """
        Суммарный объём исходящих переводов за window_hours часов.
        Используется для детекции быстрого вывода средств после ATO.
        """
        if not self.G.has_node(client_id):
            return 0.0
        cutoff = current_time - timedelta(hours=window_hours)
        total = 0.0
        for u, v, d in self.G.out_edges(client_id, data=True):
            if d.get("type") == "transfer":
                ts = d.get("timestamp")
                if ts and ts >= cutoff:
                    total += d.get("amount", 0.0)
        return total


    def summary(self) -> dict:
        """Краткая статистика графа."""
        node_types = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types = {}
        for _, _, data in self.G.edges(data=True):
            t = data.get("type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        tg = self._get_transfer_graph()
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "transfer_graph_nodes": tg.number_of_nodes(),
            "transfer_graph_edges": tg.number_of_edges(),
            "pagerank_cached_nodes": len(self._pagerank_cache),
        }