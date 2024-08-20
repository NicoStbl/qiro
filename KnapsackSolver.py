class KnapsackSolver:
    def __init__(self, maxWeight, weights, values):
        self.maxWeight = maxWeight
        self.weights = weights
        self.values = values

    def solve(self):
        # Number of items
        n = len(self.weights)

        # Initialize a 2D list to store the maximum value at each n-th item and w weight
        dp = [[0 for _ in range(self.maxWeight + 1)] for _ in range(n + 1)]

        # Build table dp[][] in bottom-up manner
        for i in range(1, n + 1):
            for w in range(1, self.maxWeight + 1):
                # If including item i-1 (0-indexed in weights) exceeds the weight, skip it
                if self.weights[i - 1] <= w:
                    dp[i][w] = max(self.values[i - 1] + dp[i - 1][w - self.weights[i - 1]], dp[i - 1][w])
                else:
                    dp[i][w] = dp[i - 1][w]

        # The last element of dp[n][maxWeight] is the maximum value that can be attained
        max_value = dp[n][self.maxWeight]

        # Backtrack to find which items are included in the optimal solution
        w = self.maxWeight
        selected_items = []

        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append([self.weights[i - 1], self.values[i-1]])  # Add this item to the list
                w -= self.weights[i - 1]

        selected_items.reverse()  # Reverse the list to get the order in which items were added

        return max_value, selected_items