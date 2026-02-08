import threading

import roslibpy

if __name__ == "__main__":
    ros = roslibpy.Ros(host="localhost", port=9090)

    done = threading.Event()

    def print_result(value):
        print("+++++++++++++++ result:", value)
        done.set()

    def print_error(value):
        print("+++++++++++++++ error:", value)
        done.set()

    def print_feedback(value):
        print("+++++++++++++++ feedback:", value)

    ros.run()

    action_cli = roslibpy.ActionClient(
        ros,
        "/joint_trajectory_controller/follow_joint_trajectory",
        "control_msgs/action/FollowJointTrajectory",
    )

    msg_data = msg_data = {
        # 顶级字段是 control_msgs/action/FollowJointTrajectory 的 Goal 消息
        "trajectory": {
            "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"],
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
            "points": [
                {
                    # JointTrajectoryPoint 的字段
                    "positions": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "velocities": [],  # 建议添加空的速度字段，以防控制器要求
                    "accelerations": [],  # 建议添加空的加速度字段
                    "effort": [],
                    "time_from_start": {"sec": 1, "nanosec": 0},
                },
                {
                    "positions": [0.0, 1.0, -1.57, -1.57, 0.2, 0.0],
                    "velocities": [],
                    "accelerations": [],
                    "effort": [],
                    "time_from_start": {"sec": 3, "nanosec": 0},
                },
            ],
        },
        # 可选的容差字段，最好明确指定为空列表或零容差
        "path_tolerance": [],
        "goal_tolerance": [],
        "goal_time_tolerance": {"sec": 0, "nanosec": 0},
    }
    print(f"++++ prepare goal {msg_data}")

    goal = roslibpy.Goal(msg_data)

    goal_id = action_cli.send_goal(
        goal,
        print_result,
        print_feedback,
        print_error,
    )

    print("+++++++= send goal id =", goal_id)

    done.wait()
    print("++++++++++++++ done")
    ros.terminate()
