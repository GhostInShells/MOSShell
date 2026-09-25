// screen-node 视觉原型: background 槽 + 浮游 meta 层 + 聚焦层.
// 交互: 点浮游图标 → 物化进 focus 槽; 点窗口 → 退回浮游层; 左上角切 layout.
import QtQuick
import QtQuick.Window

Window {
    id: root
    width: 1280
    height: 800
    visible: true
    color: "#0d1117"
    title: "screen node — QML visual demo"

    // ---- layout 状态 ----
    property string layoutName: "solo"   // solo | split
    property int maxSlots: layoutName === "solo" ? 1 : 2
    property var focused: []             // uid 列表, 顺序即槽位

    function slotRect(slot) {
        const margin = 24
        const topBar = 64
        const w = root.width
        const h = root.height
        const bodyH = h - topBar - 260   // 底部留给数字人
        if (layoutName === "solo")
            return { x: w * 0.17, y: topBar, w: w * 0.66, h: bodyH }
        const half = (w - margin * 3) / 2
        return { x: margin + slot * (half + margin), y: topBar, w: half, h: bodyH }
    }

    function toggleFocus(uid) {
        let f = focused.slice()
        const idx = f.indexOf(uid)
        if (idx >= 0) {
            f.splice(idx, 1)
        } else {
            if (f.length >= maxSlots)
                f.shift()   // 槽满: 最早的退回浮游层
            f.push(uid)
        }
        focused = f
    }

    onMaxSlotsChanged: {
        if (focused.length > maxSlots)
            focused = focused.slice(focused.length - maxSlots)
    }

    // ---- background 槽: 渐变底 + 数字人占位 (呼吸) ----
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0d1117" }
            GradientStop { position: 1.0; color: "#141a2e" }
        }
    }
    Item {
        width: 220; height: 220
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 24
        Rectangle {
            anchors.centerIn: parent
            width: 140; height: 140; radius: 70
            color: "#1f6feb"
            opacity: 0.85
            SequentialAnimation on scale {
                loops: Animation.Infinite
                NumberAnimation { to: 1.08; duration: 2400; easing.type: Easing.InOutSine }
                NumberAnimation { to: 1.0; duration: 2400; easing.type: Easing.InOutSine }
            }
        }
        Text {
            text: "ghost"
            color: "#8b949e"
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
        }
    }

    // ---- meta 单元数据 (将来源自 provide 语义 / servers://) ----
    ListModel {
        id: cells
        ListElement { uid: "term"; label: "term"; hue: "#3fb950"; msize: 64; badge: 0 }
        ListElement { uid: "blog"; label: "blog"; hue: "#d29922"; msize: 88; badge: 0 }
        ListElement { uid: "mail"; label: "mail"; hue: "#f85149"; msize: 56; badge: 3 }
        ListElement { uid: "docs"; label: "docs"; hue: "#a371f7"; msize: 72; badge: 0 }
        ListElement { uid: "chat"; label: "chat"; hue: "#58a6ff"; msize: 80; badge: 0 }
    }

    // ---- 浮游层 + 聚焦层: 同一 delegate 的两种形态 ----
    Repeater {
        model: cells
        delegate: Rectangle {
            id: cell
            property int slot: root.focused.indexOf(uid)
            property bool isFocused: slot >= 0
            property real driftX: 120 + index * 170
            property real driftY: 420 + (index % 3) * 60

            x: isFocused ? root.slotRect(slot).x : driftX
            y: isFocused ? root.slotRect(slot).y : driftY
            width: isFocused ? root.slotRect(slot).w : msize
            height: isFocused ? root.slotRect(slot).h : msize
            radius: isFocused ? 14 : msize / 2
            color: isFocused ? "#161b22" : hue
            border.color: hue
            border.width: isFocused ? 1.5 : 0
            z: isFocused ? 10 : 1

            Behavior on x { NumberAnimation { duration: 1100; easing.type: Easing.InOutCubic } }
            Behavior on y { NumberAnimation { duration: 1100; easing.type: Easing.InOutCubic } }
            Behavior on width { NumberAnimation { duration: 1100; easing.type: Easing.InOutCubic } }
            Behavior on height { NumberAnimation { duration: 1100; easing.type: Easing.InOutCubic } }
            Behavior on radius { NumberAnimation { duration: 1100 } }
            Behavior on color { ColorAnimation { duration: 500 } }

            // 漂浮: 定时换随机目标, Behavior 平滑跟随
            Timer {
                interval: 2600 + index * 640
                running: !cell.isFocused
                repeat: true
                triggeredOnStart: true
                onTriggered: {
                    cell.driftX = 60 + Math.random() * (root.width - 200)
                    cell.driftY = 100 + Math.random() * (root.height - 380)
                }
            }
            // 呼吸缩放
            SequentialAnimation on scale {
                running: !cell.isFocused
                loops: Animation.Infinite
                NumberAnimation { to: 1.1; duration: 1700 + index * 260; easing.type: Easing.InOutSine }
                NumberAnimation { to: 0.94; duration: 1700 + index * 260; easing.type: Easing.InOutSine }
            }
            onIsFocusedChanged: if (isFocused) scale = 1.0

            // 浮游态: 图标 + 红点
            Text {
                visible: !cell.isFocused
                anchors.centerIn: parent
                text: label
                color: "#0d1117"
                font.bold: true
            }
            Rectangle {
                visible: !cell.isFocused && badge > 0
                width: 20; height: 20; radius: 10
                color: "#da3633"
                border.color: "#0d1117"
                anchors.right: parent.right
                anchors.top: parent.top
                Text {
                    anchors.centerIn: parent
                    text: badge
                    color: "white"
                    font.pixelSize: 11
                    font.bold: true
                }
            }

            // 聚焦态: 窗口占位 (将来是 WebEngineView 物化点)
            Column {
                visible: cell.isFocused
                anchors.fill: parent
                anchors.margins: 16
                spacing: 10
                Text {
                    text: label + "  —  webview 将物化于此"
                    color: hue
                    font.pixelSize: 18
                    font.bold: true
                }
                Rectangle { width: parent.width; height: 1; color: "#30363d" }
                Text {
                    text: "focus 槽内容占位\n点击任意处 unfocus, 退回浮游层"
                    color: "#8b949e"
                    lineHeight: 1.4
                }
            }

            MouseArea {
                anchors.fill: parent
                onClicked: root.toggleFocus(uid)
            }
        }
    }

    // ---- 顶栏: layout 切换 ----
    Row {
        x: 24; y: 16
        spacing: 12
        z: 20
        Repeater {
            model: ["solo", "split"]
            delegate: Rectangle {
                width: 84; height: 32; radius: 16
                color: root.layoutName === modelData ? "#1f6feb" : "#21262d"
                Text { anchors.centerIn: parent; text: modelData; color: "#c9d1d9" }
                MouseArea { anchors.fill: parent; onClicked: root.layoutName = modelData }
            }
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: "layout: " + root.layoutName
                  + (root.focused.length ? "    focused: " + root.focused.join(", ") : "")
            color: "#8b949e"
        }
    }
}
