---
description: 倒数→花京院→加速レロレロ（给小孩的整活）
name: kakyoin
---

<matrix.mesh.arts:write kind="html" label="kakyoin" description="花京院">
MACRO_CDATA_START<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#000;overflow:hidden}
#scene{position:fixed;inset:0;display:grid;place-items:center}
#scene img{height:94vh;animation:pulse .26s infinite alternate ease-in-out}
@keyframes pulse{from{transform:scale(1.0) rotate(-.7deg)}to{transform:scale(1.07) rotate(.7deg)}}
#lero{position:fixed;left:0;right:0;bottom:3vh;text-align:center;font-family:system-ui;font-weight:800;font-size:8vh;color:#ffe24a;letter-spacing:.12em;text-shadow:0 0 30px rgba(255,220,0,.8)}
</style></head><body>
<div id="scene"><img src="http://127.0.0.1:20381/asset/kakyoin.jpg" alt=""></div>
<div id="lero">レロレロレロ</div>
</body></html>MACRO_CDATA_END
</matrix.mesh.arts:write>
<matrix.mesh.arts:display label="blank"/>
<all><matrix.mesh.screen:danmaku content="5" at_x="0.5" at_y="0.45" duration="1.0" size="260" color="#ff6b4a"/>五</all>
<all><matrix.mesh.screen:danmaku content="4" at_x="0.5" at_y="0.45" duration="1.0" size="260" color="#ff6b4a"/>四</all>
<all><matrix.mesh.screen:danmaku content="3" at_x="0.5" at_y="0.45" duration="1.0" size="260" color="#ff6b4a"/>三</all>
<all><matrix.mesh.screen:danmaku content="2" at_x="0.5" at_y="0.45" duration="1.0" size="260" color="#ff6b4a"/>二</all>
<all><matrix.mesh.screen:danmaku content="1" at_x="0.5" at_y="0.45" duration="1.0" size="260" color="#ff6b4a"/>一</all>
<sleep duration="0.8"/>
<all><matrix.mesh.arts:display label="kakyoin"/><say voice:dict="{'speech_rate': -30}">レロレロレロレロレロ</say></all>
<all><say voice:dict="{'speech_rate': 0}">レロレロレロレロレロ</say></all>
<all><say voice:dict="{'speech_rate': 30}">レロレロレロレロレロ</say></all>
<all><say voice:dict="{'speech_rate': 60}">レロレロレロレロレロ</say></all>
<all><say voice:dict="{'speech_rate': 85}">レロレロレロレロレロ</say></all>
<all><say voice:dict="{'speech_rate': 100}">レロレロレロレロレロ</say></all>
