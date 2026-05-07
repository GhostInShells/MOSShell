from ghoshell_moss.host.abcd.app import AppStore

__all__ = ['AppStoreREPL']


class AppStoreREPL:
    """用于在 REPL 中观测 Manifest 资源的工具集"""

    def __init__(self, apps: AppStore):
        self._apps = apps

    def list_apps(self) -> list[dict]:
        """
        展示当前环境发现的所有 apps.
        """
        app_infos = self._apps.list_apps()
        result = []
        for app_info in app_infos:
            result.append(dict(
                fullname=app_info.fullname,
                description=app_info.description,
                state=app_info.state,
                error=app_info.error,
                workspace_dir=app_info.work_directory,
                watcher=app_info.watcher.model_dump(),
            ))
        return result

    async def start_app(self, fullname: str, argument: str = '') -> str:
        """
        start app by fullname.
        if argument is empty, use default argument.
        """
        return await self._apps.start_app(fullname, argument)
