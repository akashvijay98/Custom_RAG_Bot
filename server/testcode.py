import sys
from typing import NewType, List, Dict, Set, Tuple

# Type alias for clarity
SatelliteId = NewType("SatelliteId", int)


# --- Helper functions ---
def on_satellite_reported_back(satellite_id: SatelliteId) -> None:
    print(f"SatelliteReportedBack: {satellite_id}")


# ... (other helper functions like err_duplicate_satellite would be here)

class SatelliteNetwork:
    def __init__(self):
        self.satellites: Set[SatelliteId] = set()
        self.graph: Dict[SatelliteId, List[SatelliteId]] = {}

    def satellite_connected(self, satellite_id: SatelliteId) -> None:
        self.satellites.add(satellite_id)
        self.graph[satellite_id] = []

    def relationship_established(self, satellite_id1: SatelliteId, satellite_id2: SatelliteId) -> None:
        self.graph[satellite_id1].append(satellite_id2)
        self.graph[satellite_id2].append(satellite_id1)

    def message_received(self, satellite_ids: List[SatelliteId]) -> None:
        if not satellite_ids:
            return

        # --- Phase 1: Naive Iterative Calculation of Arrival Times ---
        time_received: Dict[SatelliteId, float] = {sat_id: float('inf') for sat_id in self.satellites}
        source_map: Dict[SatelliteId, int] = {}
        initial_set = set(satellite_ids)

        for s_id in satellite_ids:
            time_received[s_id] = 0
            source_map[s_id] = -1  # -1 represents Earth

        # Loop N-1 times (N = number of satellites) to ensure propagation
        # This is the core of the naive, non-Dijkstra approach.
        for _ in range(len(self.satellites)):
            made_change = False
            # On each pass, check every satellite to see if it can offer a better time
            for s_id in self.satellites:
                if time_received[s_id] == float('inf'):
                    continue  # This satellite hasn't received the message yet

                current_time = time_received[s_id]
                source_sat = source_map.get(s_id)

                time_offset = 0
                neighbors_to_notify = sorted(self.graph.get(s_id, []))

                for neighbor in neighbors_to_notify:
                    # Apply the same notification rules as before
                    if neighbor == source_sat or (source_sat == -1 and neighbor in initial_set):
                        continue

                    time_offset += 10
                    arrival_time = current_time + time_offset

                    # If we found an earlier path to the neighbor, update it
                    if arrival_time < time_received[neighbor]:
                        time_received[neighbor] = arrival_time
                        source_map[neighbor] = s_id
                        made_change = True

            # Optimization: if a full pass makes no changes, we can stop early
            if not made_change:
                break

        # --- Phase 2: Calculate final report-back times (This logic is unchanged) ---
        reports: List[Tuple[int, SatelliteId]] = []
        for sat_id, t_rec in time_received.items():
            if t_rec == float('inf'):
                continue

            total_forwarding_time = 0
            source = source_map.get(sat_id)

            for neighbor in sorted(self.graph.get(sat_id, [])):
                if neighbor == source or (source == -1 and neighbor in initial_set):
                    continue

                attempt_start_time = t_rec + total_forwarding_time

                if time_received.get(neighbor, float('inf')) <= attempt_start_time:
                    continue

                total_forwarding_time += 10

            processing_time = 30
            final_report_time = t_rec + total_forwarding_time + processing_time
            reports.append((final_report_time, sat_id))

        # --- Phase 3: Sort and output results ---
        reports.sort()
        for _, sat_id in reports:
            on_satellite_reported_back(sat_id)


# --- Example Usage with the Sample Case ---
if __name__ == "__main__":
    network = SatelliteNetwork()
    instructions = [
        "SatelliteConnected 1", "SatelliteConnected 2", "SatelliteConnected 3",
        "SatelliteConnected 4", "SatelliteConnected 5",
        "RelationshipEstablished 1 3", "RelationshipEstablished 1 2",
        "RelationshipEstablished 2 5", "RelationshipEstablished 3 2",
        "RelationshipEstablished 3 4", "RelationshipEstablished 3 5",
        "MessageReceived 2 1 3"
    ]
    for line in instructions:
        parts = line.split()
        command = parts[0]
        args = [int(p) for p in parts[1:]]

        if command == "SatelliteConnected":
            network.satellite_connected(args[0])
        elif command == "RelationshipEstablished":
            network.relationship_established(args[0], args[1])
        elif command == "MessageReceived":
            network.message_received(args[1:])