import reflex as rx

config = rx.Config(
    app_name="moss_in_reflex",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    frontend_packages=[
    ]
)