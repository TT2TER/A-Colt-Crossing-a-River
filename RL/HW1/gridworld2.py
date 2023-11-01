MAX_iterations = 100000  # Maximum number of iterations
MAX_num = 5  # Decimal places to keep
MAX_delta = 0.0000001  # Maximum error
import random
def policy_evaluation(chess_number):
    global score
    global newscore
    endpos1 = [[0, 0], [chess_number - 1, chess_number - 1]]
    score = [[0.0 for _ in range(chess_number)] for _ in range(chess_number)]
    newscore = [[0.0 for _ in range(chess_number)] for _ in range(chess_number)]
    global iteration
    for iteration in range(MAX_iterations):
        global delta
        delta = 0.0
        if iteration <= 10 or int(iteration) % 10 == 0:
            print("Iteration", iteration, ":")
            for row in score:
                print(row)

        for i in range(chess_number):
            for j in range(chess_number):
                R = 4
                if [i, j] in endpos1:
                    score[i][j] = 0.0
                else:
                    # All four directions with equal probability
                    actions = ['Up', 'Down', 'Left', 'Right']
                    random.shuffle(actions)  # Shuffle to introduce randomness
                    total_value = 0
                    for action in actions:
                        if action == 'Up':
                            ni, nj = i - 1, j
                        elif action == 'Down':
                            ni, nj = i + 1, j
                        elif action == 'Left':
                            ni, nj = i, j - 1
                        elif action == 'Right':
                            ni, nj = i, j + 1

                        # Check if the action is valid (within the chessboard)
                        if 0 <= ni < chess_number and 0 <= nj < chess_number:
                            total_value += score[ni][nj]
                        else:
                            total_value += score[i][j]  # Stay in the same position

                    newscore[i][j] = round(total_value / 4, MAX_num)

                    if [ni, nj] in endpos1:
                        R = 3

                    delta = max(delta, abs(score[i][j] - newscore[i][j]))

        score = [row[:] for row in newscore]
        if delta < MAX_delta:
            break

def policy_improvement(chess_number):
    global optimal_policy
    policy_stable = True
    for i in range(chess_number):
        for j in range(chess_number):
            if optimal_policy[i][j] != 'End':
                old_action = optimal_policy[i][j]

                # Try all possible actions and calculate their expected values
                action_values = []
                for action in ['Up', 'Down', 'Left', 'Right']:
                    if action == 'Up':
                        ni, nj = i - 1, j
                    elif action == 'Down':
                        ni, nj = i + 1, j
                    elif action == 'Left':
                        ni, nj = i, j - 1
                    elif action == 'Right':
                        ni, nj = i, j + 1

                    # Check if the action is valid (within the chessboard)
                    if 0 <= ni < chess_number and 0 <= nj < chess_number:
                        action_values.append(score[ni][nj])
                    else:
                        action_values.append(score[i][j])  # Stay in the same position

                # Find the best action based on the expected values
                best_action = ['Up', 'Down', 'Left', 'Right'][action_values.index(max(action_values))]

                if best_action != old_action:
                    policy_stable = False  # Policy is not stable
                optimal_policy[i][j] = best_action

    return policy_stable

if __name__ == "__main__":
    chess_number = 4
    optimal_policy = [['' for _ in range(chess_number)] for _ in range(chess_number)]

    policy_stable = False
    while not policy_stable:
        policy_evaluation(chess_number)
        policy_stable = policy_improvement(chess_number)

    # After policy iteration is complete, you can print the final policy and values
    print("Final Optimal Policy:")
    for row in optimal_policy:
        print(row)
    print("Final Value Function:")
    for row in score:
        print([round(element, 3) for element in row])
