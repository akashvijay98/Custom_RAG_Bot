import sys


def maxCardCount(n, card):
    # dp[power] = max_cards_count
    dp = {0: 0}

    for c in card:
        # Create a copy to represent the "Skip" choice for all existing states
        new_dp = dp.copy()

        for power, count in dp.items():
            # Try the "Take" choice
            if power + c >= 0:
                new_power = power + c
                new_count = count + 1
                # Update if this path gives more cards for the same power
                if new_power not in new_dp or new_count > new_dp[new_power]:
                    new_dp[new_power] = new_count

        dp = new_dp

    return max(dp.values())


if __name__ == "__main__":
    try:
        line1 = sys.stdin.readline()
        if line1.strip():
            n = int(line1.strip())
            line2 = sys.stdin.readline()
            if line2.strip():
                cards = list(map(int, line2.strip().split()))
                print(maxCardCount(n, cards))
    except ValueError:
        pass