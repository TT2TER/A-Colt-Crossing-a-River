import numpy as np

def is_rotation_matrix(matrix):
    # Check if the matrix is square
    if matrix.shape != (3, 3):
        return False

    # Check if the matrix is orthogonal
    is_orthogonal = np.allclose(np.dot(matrix, matrix.T), np.eye(3),atol=1e-03)
    # print(is_orthogonal)
    if not is_orthogonal:
        return False

    # Check if the determinant is 1
    determinant = np.linalg.det(matrix)
    if not np.isclose(determinant, 1, atol=1e-03):
        return False

    return True

def get_rotation_angle(matrix):
    # Calculate the rotation angle using the trace of the matrix
    trace = np.trace(matrix)
    angle = np.arccos((trace - 1) / 2)

    return angle

# Example usage
matrix = np.array([[0.2120, 0.7743, 0.5963], [0.2120,-0.6321,0.7454], [0.9540,-0.0316,-0.2981]])
if is_rotation_matrix(matrix):
    angle = get_rotation_angle(matrix)
    print("The matrix is a rotation matrix.")
    print("Rotation angle: ", angle)
else:
    print("The matrix is not a rotation matrix.")