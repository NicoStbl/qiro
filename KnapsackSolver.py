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
                    # quantum: we might want to choose the item with the highest correlation
                    # but, that does not improve runtime or optimality
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

    def greedy(self):
        weight = 0
        value = 0
        solution = []
        items = self._orderItemsBasedOnVW()
        # take the items with the highest quantum correlation
        while len(items) > 0 and len(items[0]) and items[0][0] + weight <= self.maxWeight:
            weight += items[0][0]
            value += items[0][1]
            solution.append(items.pop(0))

        print("Greedy solution: ", solution, " with total value: ", value, " and total weight: ", weight)
        return value, weight, solution

    def greedy_with_pruning(self):
        weight = 0
        value = 0
        solution = []
        items = self._orderItemsBasedOnVW()

        # Pruning threshold (percentage of remaining capacity that should be left)
        pruning_threshold = 0.1  # 10% of maxWeight

        # Lookahead: Consider the impact of each choice on future options
        while len(items) > 0:
            # Find the best item to add, considering future potential
            best_value = -float('inf')
            best_item = None
            best_index = -1

            for i, item in enumerate(items):
                item_weight, item_value = item
                if item_weight + weight <= self.maxWeight:
                    # Check how much capacity would be left if this item were added
                    remaining_capacity = self.maxWeight - (item_weight + weight)

                    # Estimate the potential future value that can be achieved with the remaining capacity
                    future_value = item_value
                    for future_item in items[i + 1:]:
                        future_item_weight, future_item_value = future_item
                        if future_item_weight <= remaining_capacity:
                            future_value += future_item_value
                            remaining_capacity -= future_item_weight

                    # Prune the item if adding it would leave too little capacity
                    if remaining_capacity > 0 and (remaining_capacity / self.maxWeight) < pruning_threshold:
                        continue

                    # Determine if this is the best item to add
                    if future_value > best_value:
                        best_value = future_value
                        best_item = item
                        best_index = i

            # If a suitable item was found, add it to the solution
            if best_item:
                weight += best_item[0]
                value += best_item[1]
                solution.append(best_item)
                items.pop(best_index)
            else:
                break  # No more feasible items to add

        print("Unlazy Greedy solution: ", solution, " with total value: ", value, " and total weight: ", weight)
        return value, weight, solution

    def advanced_greedy(self):
        # Initialize remaining capacity (RHS)
        RHS = self.maxWeight
        solution = []

        # Initialize the list of items with (weight, value, index)
        items = [(self.weights[i], self.values[i], i) for i in range(len(self.weights))]

        # Sort items based on weight in ascending order
        items.sort(key=lambda x: x[0])

        # Calculate n_j for each item
        n_j = [RHS // item[0] for item in items]

        # Determine n_0 (largest index such that the sum of the smallest n_0 weights <= RHS)
        n_0 = 0
        cumulative_weight = 0
        for i, item in enumerate(items):
            cumulative_weight += item[0]
            if cumulative_weight <= RHS:
                n_0 = i + 1
            else:
                break

        # Calculate P_{Kj} = p_j * n_j'
        P_Kj = [item[1] * min(n_j[i], n_0) for i, item in enumerate(items)]

        # Sort items by P_{Kj} in descending order
        items_sorted_by_PKj = sorted(zip(P_Kj, items), key=lambda x: x[0], reverse=True)

        # Greedily select items based on the sorted P_{Kj} values
        total_value = 0
        total_weight = 0
        for P, item in items_sorted_by_PKj:
            if total_weight + item[0] <= RHS:
                solution.append(item[2])  # Append the index of the item
                total_weight += item[0]
                total_value += item[1]

        print("Advanced Greedy solution: ", solution, " with total value: ", total_value, " and total weight: ", total_weight)
        return total_value, total_weight, solution

    def _orderItemsBasedOnVW(self):
        # Order the items based on the value-to-weight ratio
        items = [[self.weights[i], self.values[i]] for i in range(len(self.weights))]
        items.sort(key=lambda x: x[1] / x[0], reverse=True)
        return items