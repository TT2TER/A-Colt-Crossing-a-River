MAX_iterations = 100000  # Maximum number of iterations
MAX_num = 5  # Decimal places to keep
MAX_delta = 0.0000001  # Maximum error

def chess_score(chess_number):
    global score
    global newscore
    endpos1 = [[0, 0], [chess_number - 1, chess_number - 1]]
    score = [[0.0 for _ in range(chess_number)] for _ in range(chess_number)]
    newscore = [[0.0 for _ in range(chess_number)] for _ in range(chess_number)]
    global iteration
    for iteration in range(MAX_iterations):
        global delta
        delta = 0.0
        if iteration <= 10 or int(iteration) % 10 == 0:  # To make the results clearer, print every 10 iterations after the first 10
            print("Iteration", iteration, ":")
            for row in score:
                print(row)
        for i in range(chess_number):
            for j in range(chess_number):
                R=4
                if [i, j] in endpos1:
                    score[i][j] = 0.0
                else:
                    total = 0
                    count = 0
                    for x in [-1, 1]: 
                        ni = i + x
                        if 0 <= ni < chess_number:
                            total += score[ni][j]
                            count += 1
                            if [ni, j] in endpos1:
                                R = 3
                    for y in [-1, 1]:
                        nj = j + y
                        if 0 <= nj < chess_number:
                            total += score[i][nj]
                            count += 1
                            if [i, nj] in endpos1:
                                R = 3
                    newscore[i][j] = round(((total + (4 - count) * score[i][j]) - R) / 4, MAX_num)
                    delta = max(delta, abs(score[i][j] - newscore[i][j]))

        score = [row[:] for row in newscore]
        if delta < MAX_delta:
            break

    optimal_policy = [['' for _ in range(chess_number)] for _ in range(chess_number)]
    for i in range(chess_number):
        for j in range(chess_number):
            if [i, j] in endpos1:
                optimal_policy[i][j] = 'End'
            else:
                best_actions = []
                max_neighbor = float('-inf')
                for x in [-1, 1]: 
                    ni = i + x
                    if 0 <= ni < chess_number:
                        neighbor_value = score[ni][j]
                        if neighbor_value > max_neighbor:
                            max_neighbor = neighbor_value
                            best_actions = ['Up'] if x == -1 else ['Down']
                        elif neighbor_value == max_neighbor:
                            best_actions.append('Up' if x == -1 else 'Down')
                for y in [-1, 1]:
                    nj = j + y
                    if 0 <= nj < chess_number:
                        neighbor_value = score[i][nj]
                        if neighbor_value > max_neighbor:
                            max_neighbor = neighbor_value
                            best_actions = ['Left'] if y == -1 else ['Right']
                        elif neighbor_value == max_neighbor:
                            best_actions.append('Left' if y == -1 else 'Right')
                optimal_policy[i][j] = '/'.join(best_actions)

    print("Final result after", int(iteration + 1), "iterations:")
    for row in score:
        print([round(element, 3) for element in row])
    print("Optimal Policy:")
    for row in optimal_policy:
        print(row)
    print("Calculation error:", delta)

if __name__ == "__main__":
    chess_score(4)
