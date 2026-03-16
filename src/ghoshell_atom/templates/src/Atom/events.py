from ghoshell_ghost.concepts.eventbus import EventModel
# 加载系统框架默认的 events.
from ghoshell_atom.framework.events import *

"""
Atom 全局使用的 events 声明. 
本文件里实现了 EventModel 的子类会自动加入到 Atom.event_types() 作为自解释约定. 
"""
