import time

import roslibpy

if __name__ == "__main__":
    ros = roslibpy.Ros(host="localhost", port=9090)

    def print_joint_states(value):
        print("+++++++++=", value)

    topic = roslibpy.Topic(
        ros,
        "/joint_states",
        "sensor_msgs/JointState",
    )
    topic.subscribe(print_joint_states)

    ros.run()
    time.sleep(10.0)
    ros.terminate()
