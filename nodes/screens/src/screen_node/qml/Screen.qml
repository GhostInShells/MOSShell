// Screen Node — QML compositor: background + focus + front + float.
//
// Bridge calls (open/focus/float/...) mutate JS properties; QML bindings
// drive animated transitions. Human clicks mutate state directly (instant
// feedback) AND record via bridge.human_clicked.
//
// WebEngine support: set root.engineReady = true after QtWebEngine init
// (from Python side). Until then, web slots show colored placeholders.
import QtQuick
import QtQuick.Window

Window {
    id: root
    width: 1280; height: 800
    visible: true
    color: "#0d1117"
    title: "screen"

    // ---- layout configuration ----
    readonly property int topBarH: 48
    readonly property int bottomPad: 220
    readonly property int margin: 20
    readonly property int animMs: 1100

    // Set from Python when QtWebEngine is ready.
    property bool engineReady: false

    // ---- window registry: {id: {url, label, title, badge, icon}} ----
    property var windows: ({})

    // ---- layout state ----
    property string layoutName: "solo"
    property string backgroundId: ""
    property string focusId: ""
    property var frontIds: []
    property var floatIds: []

    // ---- helpers ----
    function win(id) {
        return (windows && windows[id]) ? windows[id]
             : ({url:"", label:id, badge:0, title:""});
    }
    function badgeCount(id) { var w = win(id); return w.badge || 0; }

    function focusRect() {
        var bodyH = root.height - topBarH - bottomPad - margin;
        return { x: margin, y: topBarH + margin,
                 w: root.width - margin * 2, h: bodyH };
    }

    // ═══════════════════════════════════════════════════════════════════
    // Bridge operations — called from Python (GUI thread)
    // ═══════════════════════════════════════════════════════════════════

    function open_window(id, url, label) {
        if (windows === undefined) windows = ({});
        windows[id] = { url: url, label: label || id, title: "", badge: 0, icon: "" };
        windowsChanged();
        if (floatIds.indexOf(id) < 0) {
            floatIds = floatIds.concat([id]);
        }
        floatIdsChanged();
    }

    function close_window(id) {
        focusId = (focusId === id) ? "" : focusId;
        backgroundId = (backgroundId === id) ? "" : backgroundId;
        frontIds = frontIds.filter(function(x) { return x !== id; });
        floatIds = floatIds.filter(function(x) { return x !== id; });
        if (windows) delete windows[id];
        windowsChanged();
        focusIdChanged();
        floatIdsChanged();
    }

    function focus_window(id, slot) {
        _removeFromSlots(id);
        focusId = id;
        focusIdChanged();
        floatIdsChanged();
    }

    function front_window(id, index) {
        _removeFromSlots(id);
        var copy = frontIds.slice();
        copy.splice(typeof index === 'number' ? index : copy.length, 0, id);
        frontIds = copy;
        floatIdsChanged();
    }

    function float_window(id) {
        _removeFromSlots(id);
        if (floatIds.indexOf(id) < 0) {
            floatIds = floatIds.concat([id]);
        }
        floatIdsChanged();
    }

    function clear_slot(slot) {
        if (slot === "focus" && focusId) {
            var old = focusId;
            focusId = "";
            focusIdChanged();
            float_window(old);
        } else if (slot === "front") {
            while (frontIds.length > 0) float_window(frontIds[0]);
        }
    }

    function set_background(id) {
        backgroundId = id;
    }

    function switch_layout(name) {
        layoutName = name;
    }

    function _removeFromSlots(id) {
        if (focusId === id) focusId = "";
        frontIds = frontIds.filter(function(x) { return x !== id; });
        floatIds = floatIds.filter(function(x) { return x !== id; });
    }

    // ---- badge update from bridge signal ----
    Connections {
        target: bridge
        function onWindow_badge_changed(id, badge) {
            if (windows && windows[id]) {
                windows[id].badge = badge;
                windowsChanged();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Scene layers
    // ═══════════════════════════════════════════════════════════════════

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0d1117" }
            GradientStop { position: 1.0; color: "#141a2e" }
        }
    }

    // ---- background slot ----
    Loader {
        id: backgroundLoader
        anchors.fill: parent
        active: backgroundId !== ""
        sourceComponent: Item {
            anchors.fill: parent
            // When engineReady: WebEngineView { url: root.win(backgroundId).url }
            // Placeholder: tinted rect
            Rectangle {
                anchors.fill: parent
                color: Qt.rgba(0.06, 0.08, 0.14, 0.9)
                opacity: 0.6
            }
            Text {
                anchors.centerIn: parent
                text: "[bg] " + (root.win(root.backgroundId).label || root.backgroundId)
                    + "\n" + root.win(root.backgroundId).url
                color: "#484f58"; font.pixelSize: 14
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }

    // ---- background placeholder (no background window) ----
    Item {
        visible: backgroundId === ""
        anchors { horizontalCenter: parent.horizontalCenter
                  bottom: parent.bottom; bottomMargin: 36 }
        width: 200; height: 200
        Rectangle {
            anchors.centerIn: parent
            width: 120; height: 120; radius: 60
            color: "#1f6feb"; opacity: 0.7
            SequentialAnimation on scale {
                loops: Animation.Infinite
                NumberAnimation { to: 1.10; duration: 2400; easing.type: Easing.InOutSine }
                NumberAnimation { to: 1.0; duration: 2400; easing.type: Easing.InOutSine }
            }
        }
        Text {
            anchors { horizontalCenter: parent.horizontalCenter; bottom: parent.bottom }
            text: "ghost"; color: "#8b949e"; font.pixelSize: 14
        }
    }

    // ---- focus slot ----
    Loader {
        id: focusLoader
        active: focusId !== ""
        sourceComponent: Rectangle {
            id: focusPlaceholder
            property alias windowId: focusPlaceholder.wwid
            property string wwid: root.focusId
            color: "#161b22"
            border { color: root.win(windowId).label ? "#30363d" : "#1f6feb"; width: 2 }
            radius: 14

            Behavior on x { NumberAnimation { duration: root.animMs; easing.type: Easing.InOutCubic } }
            Behavior on y { NumberAnimation { duration: root.animMs; easing.type: Easing.InOutCubic } }
            Behavior on width { NumberAnimation { duration: root.animMs; easing.type: Easing.InOutCubic } }
            Behavior on height { NumberAnimation { duration: root.animMs; easing.type: Easing.InOutCubic } }

            Column {
                anchors.centerIn: parent
                spacing: 12
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: root.win(wwid).label || wwid
                    color: "#c9d1d9"; font.pixelSize: 24; font.bold: true
                }
                Rectangle { width: 200; height: 1; color: "#30363d"; anchors.horizontalCenter: parent.horizontalCenter }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: root.win(wwid).url
                    color: "#58a6ff"; font.pixelSize: 12
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: engineReady ? "" : "(WebEngine placeholder)"
                    color: "#484f58"; font.pixelSize: 11
                }
            }

            // Close button → float
            Rectangle {
                anchors { right: parent.right; top: parent.top; margins: 8 }
                width: 28; height: 28; radius: 14; color: "#30363d"
                z: 10
                Text { anchors.centerIn: parent; text: "x"; color: "#c9d1d9" }
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        root.float_window(wwid);
                        bridge.human_clicked(wwid, "unfocus");
                    }
                }
            }
        }

        onLoaded: {
            var r = root.focusRect();
            item.x = r.x; item.y = r.y;
            item.width = r.w; item.height = r.h;
        }
    }

    // ---- front strip ----
    Row {
        x: margin; spacing: 8
        y: focusRect().y + focusRect().h + 8
        Repeater {
            model: frontIds
            delegate: Rectangle {
                required property string modelData
                width: 180; height: 110; radius: 10
                color: "#161b22"
                border { color: "#30363d"; width: 1 }

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        root.floatIds = root.floatIds.filter(function(x) { return x !== modelData; });
                        root.focus_window(modelData, "focus");
                        bridge.human_clicked(modelData, "focus_from_front");
                    }
                }
                Text {
                    anchors.centerIn: parent
                    text: root.win(modelData).label || modelData
                    color: "#c9d1d9"; font.pixelSize: 16
                }
                // Badge
                Rectangle {
                    visible: root.badgeCount(modelData) > 0
                    anchors { right: parent.right; top: parent.top; margins: 6 }
                    width: 22; height: 22; radius: 11; color: "#da3633"
                    Text {
                        anchors.centerIn: parent
                        text: root.badgeCount(modelData)
                        color: "white"; font.pixelSize: 11; font.bold: true
                    }
                }
                // Close button
                Rectangle {
                    anchors { right: parent.right; top: parent.top; margins: 4 }
                    width: 18; height: 18; radius: 9; color: "#30363d"
                    Text { anchors.centerIn: parent; text: "x"; color: "#8b949e"; font.pixelSize: 10 }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            root.float_window(modelData);
                            bridge.human_clicked(modelData, "close_front");
                        }
                    }
                }
            }
        }
    }

    // ---- float layer ----
    Repeater {
        model: floatIds
        delegate: Item {
            id: metaItem
            required property string modelData
            property string wid: modelData
            property real driftX: 80 + (floatIds.indexOf(wid) % 6) * 120
                                   + Math.sin(floatIds.indexOf(wid) * 1.7) * 30
            property real driftY: root.height - bottomPad + 20
                                   + Math.floor(floatIds.indexOf(wid) / 6) * 80
                                   + Math.cos(floatIds.indexOf(wid) * 2.3) * 24

            x: driftX; y: driftY
            width: 64; height: 64
            z: 1

            Behavior on x { NumberAnimation { duration: root.animMs; easing.type: Easing.InOutCubic } }
            Behavior on y { NumberAnimation { duration: root.animMs; easing.type: Easing.InOutCubic } }

            Timer {
                interval: 3000 + Math.floor(Math.random() * 4000)
                running: true; repeat: true
                onTriggered: {
                    metaItem.driftX += (Math.random() - 0.5) * 80;
                    metaItem.driftY += (Math.random() - 0.5) * 60;
                    metaItem.x = metaItem.driftX;
                    metaItem.y = metaItem.driftY;
                }
            }
            SequentialAnimation on scale {
                running: true; loops: Animation.Infinite
                NumberAnimation { to: 1.08; duration: 2000; easing.type: Easing.InOutSine }
                NumberAnimation { to: 0.94; duration: 2000; easing.type: Easing.InOutSine }
            }

            Rectangle {
                anchors.fill: parent; radius: width / 2
                color: "#238636"; opacity: 0.9
                border { color: "#3fb950"; width: 1 }
            }
            Text {
                anchors.centerIn: parent
                text: root.win(wid).label || wid
                color: "#0d1117"; font.pixelSize: 11; font.bold: true
            }
            // Badge
            Rectangle {
                visible: root.badgeCount(wid) > 0
                anchors { right: parent.right; top: parent.top; margins: -2 }
                width: 22; height: 22; radius: 11
                color: "#da3633"; border { color: "#0d1117"; width: 1.5 }
                Text {
                    anchors.centerIn: parent
                    text: root.badgeCount(wid)
                    color: "white"; font.pixelSize: 11; font.bold: true
                }
            }
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    root.floatIds = root.floatIds.filter(function(x) { return x !== wid; });
                    root.focus_window(wid, "focus");
                    bridge.human_clicked(wid, "toggle_focus");
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Top bar
    // ═══════════════════════════════════════════════════════════════════
    Rectangle {
        x: 0; y: 0; width: parent.width; height: topBarH
        color: "#0d1117"; z: 20
        Row {
            anchors { left: parent.left; leftMargin: 16
                      verticalCenter: parent.verticalCenter }
            spacing: 10
            Repeater {
                model: ["solo"]
                delegate: Rectangle {
                    required property string modelData
                    width: 72; height: 28; radius: 14
                    color: root.layoutName === modelData ? "#1f6feb" : "#21262d"
                    Text {
                        anchors.centerIn: parent
                        text: modelData; color: "#c9d1d9"; font.pixelSize: 12
                    }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            root.switch_layout(modelData);
                            bridge.human_clicked("", "switch_layout:" + modelData);
                        }
                    }
                }
            }
        }
        Text {
            anchors { left: parent.left; leftMargin: 110
                      verticalCenter: parent.verticalCenter }
            text: "screen  |  layout: " + root.layoutName
                  + (focusId ? "  |  focus: " + focusId : "")
            color: "#8b949e"; font.pixelSize: 12
        }
    }
}
