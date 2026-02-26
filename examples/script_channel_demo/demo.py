from ghoshell_moss.transports import script_channel

script_channel.run(
    "examples/script_channel_demo/target_script.py",
    name="demo_script",
    interactive=True,
)
