import asyncio

from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProxy
from ghoshell_moss_contrib.gui.image_viewer import SimpleImageViewer, run_img_viewer
from ghoshell_moss.message.contents import Base64Image

if __name__ == "__main__":
    # 测试专用.
    proxy = ZMQChannelProxy(
        name="vision",
        address="tcp://127.0.0.1:5557",
    )

    def callback(viewer: SimpleImageViewer):

        async def main():
            async with proxy.bootstrap() as broker:
                await broker.wait_connected()
                while True:
                    await asyncio.sleep(2)
                    if not proxy.is_running():
                        continue
                    await proxy.broker.refresh_meta()
                    meta = proxy.broker.meta()
                    for msg in meta.context:
                        for ct in msg.contents:
                            if i := Base64Image.from_content(ct):
                                viewer.set_pil_image(i.to_pil_image())

        asyncio.run(main())

    run_img_viewer(callback)
