/* Badge API interception — inject into WebEngineView pages.

   Intercepts navigator.setAppBadge() / navigator.clearAppBadge() so that
   web-standard badge calls are forwarded to the screen bridge via QWebChannel,
   rather than going to the OS (where Qt can't read them).

   The page uses the standard Badging API — no knowledge of screen internals.
   Screen reads the badge value and drives both the QML meta item (for humans)
   and the peek bucket (for models).

   Usage in QML WebEngineView:
       WebEngineView {
           userScripts: [
               WebEngineScript {
                   source: "file:///path/to/badge_intercept.js"
                   injectionPoint: WebEngineScript.DocumentReady
               }
           ]
       }
*/

(function() {
    'use strict';

    // Store original badge value for getter simulation.
    var _badge = null;

    // Override setAppBadge.
    var _origSet = navigator.setAppBadge;
    navigator.setAppBadge = function(contents) {
        var value = contents === undefined ? 0 : (typeof contents === 'number' ? contents : 0);
        _badge = value;

        // Forward to Qt bridge if QWebChannel is available.
        if (typeof qt !== 'undefined' && qt.webChannelTransport) {
            try {
                if (typeof bridge !== 'undefined' && bridge.web_badge_changed) {
                    bridge.web_badge_changed(window.__screen_window_id || '', value);
                }
            } catch (e) {
                // Silently ignore — badge display is best-effort.
            }
        }

        // Also try native for other consumers (PWA, OS dock).
        if (_origSet) {
            return _origSet.call(navigator, contents);
        }
        return Promise.resolve();
    };

    // Override clearAppBadge.
    var _origClear = navigator.clearAppBadge;
    navigator.clearAppBadge = function() {
        _badge = null;

        if (typeof qt !== 'undefined' && qt.webChannelTransport) {
            try {
                if (typeof bridge !== 'undefined' && bridge.web_badge_changed) {
                    bridge.web_badge_changed(window.__screen_window_id || '', 0);
                }
            } catch (e) {}
        }

        if (_origClear) {
            return _origClear.call(navigator);
        }
        return Promise.resolve();
    };

    // Also patch document.title changes as a fallback badge source.
    // Pages that write "(3) Inbox" to the title will emit a badge event.
    var _origTitleDescriptor = Object.getOwnPropertyDescriptor(document, 'title');
    if (_origTitleDescriptor && _origTitleDescriptor.configurable) {
        var _title = document.title;
        Object.defineProperty(document, 'title', {
            get: function() { return _title; },
            set: function(value) {
                _title = value;
                // Try to extract badge number from "(N) " prefix.
                var m = value.match(/^\((\d+)\)\s/);
                if (m) {
                    var badge = parseInt(m[1], 10);
                    if (typeof navigator.setAppBadge === 'function') {
                        navigator.setAppBadge(badge);
                    }
                }
            },
            configurable: true
        });
    }
})();
