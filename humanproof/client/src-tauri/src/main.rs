// HumanProof desktop shell. The window only ever loads the bundled web app;
// it exposes no custom commands or plugins to that page (smallest possible
// native attack surface). The one platform tweak: on Linux, WebKitGTK needs
// camera/microphone capture enabled explicitly.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .setup(|_app| {
            #[cfg(target_os = "linux")]
            {
                use tauri::Manager;
                let window = _app.get_webview_window("main").expect("main window");
                window.with_webview(|wv| {
                    use webkit2gtk::glib::prelude::ObjectExt;
                    use webkit2gtk::{PermissionRequestExt, SettingsExt, UserMediaPermissionRequest, WebViewExt};
                    let view = wv.inner();
                    if let Some(settings) = WebViewExt::settings(&view) {
                        settings.set_enable_media_stream(true);
                        settings.set_enable_webaudio(true);
                    }
                    // Grant camera/mic to our own bundled page; deny every other permission kind.
                    view.connect_permission_request(|_, req| {
                        if req.is::<UserMediaPermissionRequest>() {
                            req.allow();
                        } else {
                            req.deny();
                        }
                        true
                    });
                })?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running HumanProof");
}
