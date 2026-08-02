/* Inject window.__screen_window_id into loaded pages.

   Runs at DocumentCreation (before any page JS). Defines the property
   as non-enumerable and writable. The actual ID value is set by the
   QML WebEngineView's onLoadingChanged handler via runJavaScript().

   badge_intercept.js reads this value to include the window ID in
   bridge.web_badge_changed() calls.
*/

(function() {
    'use strict';

    if (typeof window.__screen_window_id === 'undefined') {
        Object.defineProperty(window, '__screen_window_id', {
            value: '',
            writable: true,
            configurable: false,
            enumerable: false
        });
    }
})();
