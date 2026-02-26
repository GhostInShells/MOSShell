import asyncio
import logging

from ghoshell_moss.message.contents import Base64Image
from ghoshell_moss.transports.script_channel.script_channel import ScriptChannelProxy
from ghoshell_moss_contrib.gui.image_viewer import SimpleImageViewer, run_img_viewer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    _setup_logging()

    proxy = ScriptChannelProxy(
        name="vision",
        provider_launcher="examples/vision_exam_script_channel/vision_provider.py",
        # provider_target=None -> directly run the provider script
        logger=logging.getLogger("vision_proxy"),
    )

    def callback(viewer: SimpleImageViewer):

        async def main():
            async with proxy.bootstrap() as broker:
                await broker.wait_connected()
                while True:
                    await asyncio.sleep(2)
                    if not broker.is_running():
                        continue
                    await broker.refresh_metas()
                    meta = broker.own_meta()
                    for msg in meta.context:
                        for ct in msg.contents:
                            if i := Base64Image.from_content(ct):
                                viewer.set_pil_image(i.to_pil_image())

        asyncio.run(main())

    run_img_viewer(callback)
