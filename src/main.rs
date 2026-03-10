#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use desktop::TranslateDesktopApp;

mod desktop;
mod translate;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 760.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title("Translate Desktop"),
        ..Default::default()
    };

    eframe::run_native(
        "Translate Desktop",
        options,
        Box::new(|cc| Ok(Box::new(TranslateDesktopApp::new(cc)))),
    )
}
