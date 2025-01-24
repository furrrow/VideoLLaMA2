import time
import numpy as np
import cv2

def compute_trajectory(v, w, dt, steps):
    x, y, theta = 0, 0, 0  # Initial pose
    trajectory = []
    for _ in range(steps):
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        trajectory.append([x, y, 0])
        # print(x,y, v,theta,dt )
    return np.array(trajectory)


def project_to_image(trajectory, K, R, t):
    trajectory_cam = (R @ trajectory.T + t).T

    # Project points into the image frame
    points_2d = K @ trajectory_cam.T

    points_2d /= points_2d[2]
    return points_2d[:2].T

def trajectory_image(K, MODE, R, alpha, background, color_list, dt, height, image, omega_list, steps, vel, width):
    t0 = np.array([[-0.06], [-1.65], [0]])
    t1 = np.array([[-1.06], [-1.65], [0]])  # Translation vector (moving the COM)
    t2 = np.array([[0.94], [-1.65], [0]])  # Translation vector (moving the COM)
    for i, w in enumerate(omega_list):
        select_color = color_list[i % len(color_list)]
        # print(f"velocity: {vel}, w: {w}, color: {select_color}")
        trajectory_robot = compute_trajectory(vel, w, dt, steps)

        # print(f"trajectory: \n {trajectory_robot}")
        # points_2d_line = project_to_image(trajectory_robot, K, R, t0)
        points_2d = project_to_image(trajectory_robot, K, R, t1)
        points_2d_1 = project_to_image(trajectory_robot, K, R, t2)

        # points_2d_line = np.array([[width, height]]) - points_2d_line
        points_2d = np.array([[width, height]]) - points_2d
        points_2d_1 = np.array([[width, height]]) - points_2d_1
        # Load a sample camera image

        # valid_points_line = []
        valid_points = []
        valid_points_1 = []
        for point in points_2d:
            u, v = int(point[0]), int(point[1])
            if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
                valid_points.append((u, v))
        for point in points_2d_1:
            u, v = int(point[0]), int(point[1])
            if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
                valid_points_1.append((u, v))
        # for point in points_2d_line:
        #     u, v = int(point[0]), int(point[1])
        #     if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
        #         valid_points_line.append((u, v))

        if len(valid_points) > 1:
            valid_points = np.array(valid_points, dtype=np.int32)
            if MODE == "poly":
                valid_points = np.concatenate((valid_points, np.array(valid_points_1[::-1])))
                valid_points = valid_points.reshape((-1, 1, 2))

            # print(f"trajectory coordinates: {valid_points}")

            if MODE == "lines":
                cv2.polylines(image, [valid_points], isClosed=False, color=select_color, thickness=5)  # arc
            if MODE == "points":
                for center in valid_points:
                    cv2.circle(image, center, radius=1, color=select_color, thickness=5)  # dots
            if MODE == "poly":
                cv2.fillPoly(image, [valid_points], select_color)

    image = cv2.addWeighted(background, alpha, image, 1 - alpha, 0)
    return image

def main():
    # constraints
    # Camera 395 parameters (example values)
    fx, fy = 721.5 , 721.5  # Focal lengths
    cx, cy = 609.56, 172.86  # image center
    width,height = 1242, 375

    # Camera matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])


    # Camera rotaion + translation

    R = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    t = np.array([[0], [-0.2], [0]])  # Translation vector

    # Parameters
    vel = 25
    omega_list = [-1, -0.5, 0, 0.5, 1]
    dt = 0.01
    steps = 100
    color_list = [
        (0, 0, 255), # Red
        (0, 255, 0), # Green
        (255, 0, 0), # Blue
    ]

    # image_path = "/home/jim/Downloads/VLM_images/image_000007.png"
    image_path = "/home/jim/Downloads/VLM_images/image_000253.png"
    image = cv2.imread(image_path)
    background = image.copy()
    MODE = "poly"  # select lines, points, or poly
    alpha = 0.5 # transparency param for poly mode
    # image = cv2.resize(image, (1280,720))

    image = trajectory_image(K, MODE, R, alpha, background, color_list, dt, height, image, omega_list, steps, vel,
                             width)

    # Display the result
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- total time: %s seconds ---" % (time.time() - start_time))