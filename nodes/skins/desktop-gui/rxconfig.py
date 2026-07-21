import reflex as rx

config = rx.Config(
    app_name="ghoshell_desktop_gui",
    env_file=".env",
    plugins=[
        rx.plugins.SitemapPlugin(),
    ],
)
