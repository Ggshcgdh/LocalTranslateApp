use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align, Button, Color32, Context, CornerRadius, Frame, Layout, Margin, Rect, RichText,
    ScrollArea, Sense, Stroke, TextEdit, Vec2,
};

use crate::translate::{TargetLang, Translator};
use crate::translate::StartupProgress;

pub struct TranslateDesktopApp {
    source_text: String,
    translated_text: String,
    direction: TranslationDirection,
    is_translating: bool,
    startup_started_at: Instant,
    translation_started_at: Option<Instant>,
    status: WorkerStatus,
    last_error: Option<String>,
    startup_progress: Option<StartupProgress>,
    worker: WorkerHandle,
}

impl TranslateDesktopApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_theme(&cc.egui_ctx);

        Self {
            source_text: String::new(),
            translated_text: String::new(),
            direction: TranslationDirection::EnToTr,
            is_translating: false,
            startup_started_at: Instant::now(),
            translation_started_at: None,
            status: WorkerStatus::Loading,
            last_error: None,
            startup_progress: None,
            worker: WorkerHandle::spawn(),
        }
    }

    fn poll_worker(&mut self) {
        loop {
            match self.worker.events.try_recv() {
                Ok(WorkerEvent::Ready { device_label }) => {
                    self.status = WorkerStatus::Ready { device_label };
                    self.last_error = None;
                    self.startup_progress = Some(StartupProgress {
                        message: "Runtime is ready.".to_owned(),
                        progress: 1.0,
                    });
                }
                Ok(WorkerEvent::StartupProgress(progress)) => {
                    self.startup_progress = Some(progress);
                }
                Ok(WorkerEvent::TranslationCompleted(result)) => {
                    self.is_translating = false;
                    self.translation_started_at = None;
                    match result {
                        Ok(text) => {
                            self.translated_text = text;
                            self.last_error = None;
                        }
                        Err(error) => {
                            self.last_error = Some(error);
                        }
                    }
                }
                Ok(WorkerEvent::StartupFailed(error)) => {
                    self.status = WorkerStatus::Failed;
                    self.is_translating = false;
                    self.translation_started_at = None;
                    self.last_error = Some(error);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.status = WorkerStatus::Failed;
                    self.is_translating = false;
                    self.translation_started_at = None;
                    if self.last_error.is_none() {
                        self.last_error =
                            Some("The background translation worker closed unexpectedly.".into());
                    }
                    break;
                }
            }
        }
    }

    fn can_submit(&self) -> bool {
        self.status.is_ready() && !self.is_translating && !self.source_text.trim().is_empty()
    }

    fn set_direction(&mut self, direction: TranslationDirection) {
        if self.direction == direction {
            return;
        }

        self.direction = direction;
        self.translated_text.clear();
        self.last_error = None;
    }

    fn submit_translation(&mut self) {
        if !self.can_submit() {
            return;
        }

        self.is_translating = true;
        self.translation_started_at = Some(Instant::now());
        self.last_error = None;
        self.translated_text.clear();

        let command = WorkerCommand::Translate {
            text: self.source_text.clone(),
            target: self.direction.target_lang(),
        };

        if let Err(error) = self.worker.commands.send(command) {
            self.is_translating = false;
            self.translation_started_at = None;
            self.last_error = Some(format!(
                "The translation request could not be sent: {error}"
            ));
        }
    }

    fn render_header(&mut self, ui: &mut egui::Ui) {
        let compact = safe_available_width(ui) < 980.0;

        card_frame(tinted_surface_color()).show(ui, |ui| {
            ui.set_min_width(ui.available_width());

            if compact {
                self.render_header_copy(ui);
                ui.add_space(8.0);
                self.render_direction_selector(ui);
                ui.add_space(8.0);
                self.render_runtime_summary(ui);
            } else {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        self.render_header_copy(ui);
                        ui.add_space(8.0);
                        self.render_direction_selector(ui);
                    });

                    ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                        ui.vertical(|ui| {
                            ui.set_max_width(328.0);
                            self.render_runtime_summary(ui);
                        });
                    });
                });
            }

            if self.active_elapsed().is_some() {
                ui.add_space(10.0);
                self.render_header_activity(ui);
            }

            ui.add_space(8.0);
            ui.label(
                RichText::new(
                    "Set TRANSLATE_DEVICE=cpu to force CPU execution. AMD, NVIDIA, and Intel GPUs are detected through DirectML when available.",
                )
                .small()
                .color(text_muted_color()),
            );
            ui.add_space(4.0);
        });
    }

    fn render_header_copy(&self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("TRANSLATION WORKSPACE")
                .small()
                .strong()
                .color(accent_color()),
        );
        ui.add_space(2.0);
        ui.label(
            RichText::new("Translate Desktop")
                .size(26.0)
                .strong()
                .color(text_primary_color()),
        );
        ui.add_space(2.0);
        ui.label(
            RichText::new(
                "Review source text, run the translation agent, and copy a polished final result.",
            )
            .color(text_muted_color()),
        );
    }

    fn render_runtime_summary(&self, ui: &mut egui::Ui) {
        let (badge_label, badge_fill, badge_stroke, badge_text, detail) = match &self.status {
            WorkerStatus::Loading => (
                "Runtime starting",
                Color32::from_rgb(232, 239, 246),
                Stroke::new(1.0, Color32::from_rgb(193, 206, 220)),
                accent_color(),
                self.startup_progress
                    .as_ref()
                    .map(|progress| progress.message.clone())
                    .unwrap_or_else(|| {
                        "Loading model assets and selecting the best execution device.".to_owned()
                    }),
            ),
            WorkerStatus::Ready { device_label } => (
                "Agent ready",
                Color32::from_rgb(232, 241, 236),
                Stroke::new(1.0, Color32::from_rgb(188, 209, 195)),
                Color32::from_rgb(34, 92, 57),
                format!("Runtime: {device_label}"),
            ),
            WorkerStatus::Failed => (
                "Runtime offline",
                Color32::from_rgb(248, 236, 236),
                Stroke::new(1.0, Color32::from_rgb(222, 190, 190)),
                Color32::from_rgb(146, 54, 54),
                "The runtime could not be started.".to_owned(),
            ),
        };

        pill_frame(badge_fill, badge_stroke).show(ui, |ui| {
            ui.label(
                RichText::new(badge_label)
                    .small()
                    .strong()
                    .color(badge_text),
            );
        });

        ui.add_space(4.0);
        ui.label(RichText::new(detail).small().color(text_muted_color()));
    }

    fn render_direction_selector(&mut self, ui: &mut egui::Ui) {
        pill_frame(subtle_surface_color(), Stroke::new(1.0, border_color())).show(ui, |ui| {
            ui.horizontal(|ui| {
                self.render_direction_option(
                    ui,
                    TranslationDirection::EnToTr,
                    "English -> Turkish",
                );
                self.render_direction_option(
                    ui,
                    TranslationDirection::TrToEn,
                    "Turkish -> English",
                );
            });
        });
    }

    fn render_direction_option(
        &mut self,
        ui: &mut egui::Ui,
        direction: TranslationDirection,
        label: &str,
    ) {
        let selected = self.direction == direction;
        let fill = if selected {
            accent_color()
        } else {
            card_color()
        };
        let stroke = if selected {
            Stroke::new(1.0, accent_color())
        } else {
            Stroke::new(1.0, border_color())
        };
        let text_color = if selected {
            Color32::WHITE
        } else {
            text_primary_color()
        };

        let button = Button::new(RichText::new(label).color(text_color))
            .fill(fill)
            .stroke(stroke)
            .corner_radius(8)
            .min_size(Vec2::new(176.0, 38.0));

        if ui.add_enabled(!self.is_translating, button).clicked() {
            self.set_direction(direction);
        }
    }

    fn render_header_activity(&self, ui: &mut egui::Ui) {
        let Some(elapsed) = self.active_elapsed() else {
            return;
        };

        let (title, detail) = self.activity_copy(elapsed);

        ui.allocate_ui_with_layout(
            Vec2::new(safe_available_width(ui), 0.0),
            Layout::top_down(Align::Min),
            |ui| {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.add_space(6.0);

                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new(title)
                                .small()
                                .strong()
                                .color(text_primary_color()),
                        );
                        ui.label(RichText::new(detail).small().color(text_muted_color()));
                    });

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format_elapsed(elapsed))
                                .small()
                                .strong()
                                .monospace()
                                .color(accent_color()),
                        );
                    });
                });

                ui.add_space(8.0);
                if self.is_translating {
                    self.render_marquee_bar(ui, 6.0);
                } else if let Some(progress) = self.startup_progress_fraction() {
                    self.render_progress_bar(ui, 8.0, progress);
                } else {
                    self.render_marquee_bar(ui, 6.0);
                }
            },
        );
    }

    fn render_translate_button(&self, ui: &mut egui::Ui) -> bool {
        let button = Button::new(
            RichText::new("Convert")
                .size(14.0)
                .strong()
                .color(Color32::WHITE),
        )
        .fill(accent_color())
        .stroke(Stroke::new(1.0, accent_color()))
        .corner_radius(10)
        .min_size(Vec2::new(124.0, 38.0));

        ui.add_enabled(self.can_submit(), button).clicked()
    }

    fn render_error_banner(&self, ui: &mut egui::Ui, error: &str) {
        card_frame(Color32::from_rgb(250, 241, 241)).show(ui, |ui| {
            ui.label(
                RichText::new("Attention required")
                    .small()
                    .strong()
                    .color(Color32::from_rgb(146, 54, 54)),
            );
            ui.add_space(4.0);
            ui.label(RichText::new(error).color(Color32::from_rgb(122, 53, 53)));
        });
    }

    fn render_text_panels(&mut self, ui: &mut egui::Ui) {
        let available_width = safe_available_width(ui);
        let wide_layout = available_width > 1_040.0;
        let content_height = (ui.available_height() - 28.0).max(0.0);

        if wide_layout {
            let spacing = ui.spacing().item_spacing.x;
            let panel_width = ((available_width - spacing) / 2.0).max(280.0);
            let panel_height = available_or(content_height, 420.0);

            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(
                    Vec2::new(panel_width, panel_height),
                    Layout::top_down(Align::Min),
                    |ui| self.render_source_panel(ui, panel_height),
                );
                ui.allocate_ui_with_layout(
                    Vec2::new(panel_width, panel_height),
                    Layout::top_down(Align::Min),
                    |ui| self.render_result_panel(ui, panel_height),
                );
            });
        } else {
            let panel_height = available_or((content_height - 12.0).max(0.0) / 2.0, 240.0);

            self.render_source_panel(ui, panel_height);
            ui.add_space(12.0);
            self.render_result_panel(ui, panel_height);
        }
    }

    fn render_source_panel(&mut self, ui: &mut egui::Ui, panel_height: f32) {
        card_frame(card_color()).show(ui, |ui| {
            ui.set_height(panel_content_height(panel_height));
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Source text").strong().size(15.0));
                    ui.label(
                        RichText::new(format!("({})", self.direction.source_description()))
                            .small()
                            .color(text_muted_color()),
                    );
                });
                ui.add_space(10.0);

                let editor_height = panel_editor_height(ui);
                let editor = TextEdit::multiline(&mut self.source_text)
                    .id_salt("source_editor")
                    .desired_rows(18)
                    .desired_width(f32::INFINITY)
                    .margin(Margin::ZERO)
                    .hint_text(self.direction.source_hint())
                    .background_color(card_color())
                    .frame(false);

                render_scrollable_text_editor(
                    ui,
                    editor_height,
                    "source_scroll",
                    card_color(),
                    !self.is_translating && self.status.is_ready(),
                    editor,
                );

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Large documents stay contained inside this editor.")
                            .small()
                            .color(text_muted_color()),
                    );

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if self.render_translate_button(ui) {
                            self.submit_translation();
                        }
                    });
                });
            });
        });
    }

    fn render_result_panel(&mut self, ui: &mut egui::Ui, panel_height: f32) {
        card_frame(card_color()).show(ui, |ui| {
            ui.set_height(panel_content_height(panel_height));
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Translated output").strong().size(15.0));
                    ui.label(
                        RichText::new(format!("({})", self.direction.target_description()))
                            .small()
                            .color(text_muted_color()),
                    );
                });
                ui.add_space(10.0);

                let editor_height = panel_editor_height(ui);
                let result_editor = TextEdit::multiline(&mut self.translated_text)
                    .id_salt("result_editor")
                    .desired_rows(18)
                    .desired_width(f32::INFINITY)
                    .margin(Margin::ZERO)
                    .hint_text("The translation output will appear here.")
                    .background_color(subtle_surface_color())
                    .frame(false)
                    .interactive(false);

                render_scrollable_text_editor(
                    ui,
                    editor_height,
                    "result_scroll",
                    subtle_surface_color(),
                    true,
                    result_editor,
                );
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Read-only output. Use the copy action when you are ready.")
                            .small()
                            .color(text_muted_color()),
                    );

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let button = Button::new("Copy output")
                            .fill(subtle_surface_color())
                            .stroke(Stroke::new(1.0, border_color()))
                            .corner_radius(8)
                            .min_size(Vec2::new(116.0, 36.0));

                        if ui
                            .add_enabled(!self.translated_text.is_empty(), button)
                            .clicked()
                        {
                            ui.ctx().copy_text(self.translated_text.clone());
                        }
                    });
                });
            });
        });
    }

    fn active_elapsed(&self) -> Option<Duration> {
        if self.is_translating {
            self.translation_started_at
                .map(|started_at| started_at.elapsed())
        } else if matches!(self.status, WorkerStatus::Loading) {
            Some(self.startup_started_at.elapsed())
        } else {
            None
        }
    }

    fn activity_copy(&self, elapsed: Duration) -> (String, String) {
        if self.is_translating {
            const PHASES: [&str; 4] = [
                "Inspecting sentence structure.",
                "Drafting the translation pass.",
                "Refining terminology and tone.",
                "Preparing the final output.",
            ];

            let phase_index = ((elapsed.as_secs() / 3) as usize) % PHASES.len();
            (
                "Translation agent active".to_owned(),
                PHASES[phase_index].to_owned(),
            )
        } else {
            (
                "Preparing the translation runtime".to_owned(),
                self.startup_progress
                    .as_ref()
                    .map(|progress| progress.message.clone())
                    .unwrap_or_else(|| {
                        "Loading model assets and selecting the best execution device.".to_owned()
                    }),
            )
        }
    }
}

impl eframe::App for TranslateDesktopApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.poll_worker();

        if self.is_translating || matches!(self.status, WorkerStatus::Loading) {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_header(ui);
            ui.add_space(14.0);

            if let Some(error) = &self.last_error {
                self.render_error_banner(ui, error);
                ui.add_space(14.0);
            }

            self.render_text_panels(ui);
        });
    }
}

impl Drop for TranslateDesktopApp {
    fn drop(&mut self) {
        let _ = self.worker.commands.send(WorkerCommand::Shutdown);
    }
}

impl TranslateDesktopApp {
    fn activity_progress(&self) -> f32 {
        let Some(elapsed) = self.active_elapsed() else {
            return 0.0;
        };

        let cycle = 1.35;
        (elapsed.as_secs_f32() % cycle) / cycle
    }

    fn startup_progress_fraction(&self) -> Option<f32> {
        if !matches!(self.status, WorkerStatus::Loading) {
            return None;
        }

        self.startup_progress
            .as_ref()
            .map(|progress| progress.progress.clamp(0.0, 1.0))
    }

    fn render_marquee_bar(&self, ui: &mut egui::Ui, height: f32) {
        let width = safe_available_width(ui);
        let (track_rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
        let track_radius = CornerRadius::same((height / 2.0).round() as u8);
        let segment_width = (track_rect.width() * 0.18).clamp(92.0, 148.0);
        let travel = track_rect.width() + segment_width;
        let segment_left = track_rect.left() - segment_width + travel * self.activity_progress();
        let segment_rect = Rect::from_min_size(
            egui::pos2(segment_left, track_rect.top()),
            Vec2::new(segment_width, track_rect.height()),
        )
        .intersect(track_rect);

        ui.painter()
            .rect_filled(track_rect, track_radius, Color32::from_rgb(223, 229, 236));

        if segment_rect.width() > 0.0 {
            ui.painter()
                .rect_filled(segment_rect, track_radius, accent_color());
        }
    }

    fn render_progress_bar(&self, ui: &mut egui::Ui, height: f32, progress: f32) {
        let width = safe_available_width(ui);
        let (track_rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
        let track_radius = CornerRadius::same((height / 2.0).round() as u8);

        ui.painter()
            .rect_filled(track_rect, track_radius, Color32::from_rgb(223, 229, 236));

        let fill_width = track_rect.width() * progress.clamp(0.0, 1.0);
        if fill_width > 0.0 {
            let fill_rect = Rect::from_min_size(
                egui::pos2(track_rect.left(), track_rect.top()),
                Vec2::new(fill_width, track_rect.height()),
            );
            ui.painter()
                .rect_filled(fill_rect, track_radius, accent_color());
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TranslationDirection {
    TrToEn,
    EnToTr,
}

impl TranslationDirection {
    fn source_description(self) -> &'static str {
        match self {
            Self::TrToEn => "Turkish input",
            Self::EnToTr => "English input",
        }
    }

    fn target_description(self) -> &'static str {
        match self {
            Self::TrToEn => "English output",
            Self::EnToTr => "Turkish output",
        }
    }

    fn source_hint(self) -> &'static str {
        match self {
            Self::TrToEn => "Enter the Turkish text that should be translated.",
            Self::EnToTr => "Enter the English text that should be translated.",
        }
    }

    fn target_lang(self) -> TargetLang {
        match self {
            Self::TrToEn => TargetLang::En,
            Self::EnToTr => TargetLang::Tr,
        }
    }
}

enum WorkerStatus {
    Loading,
    Ready { device_label: String },
    Failed,
}

impl WorkerStatus {
    fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }
}

struct WorkerHandle {
    commands: Sender<WorkerCommand>,
    events: Receiver<WorkerEvent>,
}

impl WorkerHandle {
    fn spawn() -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();

        thread::spawn(move || {
            if let Err(panic_message) = run_worker_loop(command_rx, event_tx.clone()) {
                let _ = event_tx.send(WorkerEvent::StartupFailed(panic_message));
            }
        });

        Self {
            commands: command_tx,
            events: event_rx,
        }
    }
}

enum WorkerCommand {
    Translate { text: String, target: TargetLang },
    Shutdown,
}

enum WorkerEvent {
    StartupProgress(StartupProgress),
    Ready { device_label: String },
    TranslationCompleted(Result<String, String>),
    StartupFailed(String),
}

fn run_worker_loop(
    commands: Receiver<WorkerCommand>,
    events: Sender<WorkerEvent>,
) -> Result<(), String> {
    std::panic::catch_unwind(|| worker_loop(commands, events)).map_err(panic_payload_to_string)
}

fn worker_loop(commands: Receiver<WorkerCommand>, events: Sender<WorkerEvent>) {
    let mut translator = match Translator::new_with_progress(|progress| {
        let _ = events.send(WorkerEvent::StartupProgress(progress));
    }) {
        Ok(translator) => {
            let _ = events.send(WorkerEvent::Ready {
                device_label: translator.device_label().to_owned(),
            });
            translator
        }
        Err(error) => {
            let detailed = format!("{error:#}");
            eprintln!("translator startup failed: {detailed}");
            let _ = events.send(WorkerEvent::StartupFailed(detailed));
            return;
        }
    };

    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Translate { text, target } => {
                let result = translator
                    .translate(&text, target)
                    .map_err(|error| format!("{error:#}"));

                if events
                    .send(WorkerEvent::TranslationCompleted(result))
                    .is_err()
                {
                    return;
                }
            }
            WorkerCommand::Shutdown => return,
        }
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("The background translation worker panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("The background translation worker panicked: {message}")
    } else {
        "The background translation worker panicked for an unknown reason.".into()
    }
}

fn configure_theme(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(14.0, 14.0);
    style.spacing.button_padding = Vec2::new(16.0, 10.0);
    style.spacing.indent = 18.0;
    style.visuals = egui::Visuals::light();
    style.visuals.window_fill = canvas_color();
    style.visuals.panel_fill = canvas_color();
    style.visuals.extreme_bg_color = card_color();
    style.visuals.text_edit_bg_color = Some(card_color());
    style.visuals.widgets.noninteractive.bg_fill = card_color();
    style.visuals.widgets.noninteractive.weak_bg_fill = card_color();
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, border_color());
    style.visuals.widgets.noninteractive.fg_stroke.color = text_primary_color();
    style.visuals.widgets.noninteractive.corner_radius = CornerRadius::same(10);
    style.visuals.widgets.inactive.bg_fill = card_color();
    style.visuals.widgets.inactive.weak_bg_fill = card_color();
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, border_color());
    style.visuals.widgets.inactive.fg_stroke.color = text_primary_color();
    style.visuals.widgets.inactive.corner_radius = CornerRadius::same(10);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(245, 248, 251);
    style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(245, 248, 251);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent_color());
    style.visuals.widgets.hovered.fg_stroke.color = text_primary_color();
    style.visuals.widgets.hovered.corner_radius = CornerRadius::same(10);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(236, 241, 246);
    style.visuals.widgets.active.weak_bg_fill = Color32::from_rgb(236, 241, 246);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, accent_color());
    style.visuals.widgets.active.fg_stroke.color = text_primary_color();
    style.visuals.widgets.active.corner_radius = CornerRadius::same(10);
    style.visuals.widgets.open = style.visuals.widgets.active;
    style.visuals.selection.bg_fill = accent_color();
    style.visuals.selection.stroke.color = Color32::WHITE;
    style.visuals.hyperlink_color = accent_color();

    ctx.set_style(style);
}

fn card_frame(fill: Color32) -> Frame {
    Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.0, border_color()))
        .corner_radius(12)
        .inner_margin(Margin::symmetric(18, 16))
}

fn pill_frame(fill: Color32, stroke: Stroke) -> Frame {
    Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(8)
        .inner_margin(Margin::symmetric(10, 6))
}

fn format_elapsed(elapsed: Duration) -> String {
    let total_seconds = elapsed.as_secs();
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;

    if minutes == 0 {
        format!("00:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn available_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn safe_available_width(ui: &egui::Ui) -> f32 {
    let width = ui.available_width();
    if width.is_finite() && width > 0.0 {
        width
    } else {
        600.0
    }
}

fn panel_editor_height(ui: &egui::Ui) -> f32 {
    (ui.available_height() - 54.0).max(132.0)
}

fn panel_content_height(panel_height: f32) -> f32 {
    (panel_height - 32.0).max(0.0)
}

fn scroll_editor_viewport_height(editor_height: f32) -> f32 {
    (editor_height - 22.0).max(0.0)
}

fn render_scrollable_text_editor(
    ui: &mut egui::Ui,
    editor_height: f32,
    scroll_id: &'static str,
    fill: Color32,
    enabled: bool,
    editor: TextEdit<'_>,
) -> egui::Response {
    let editor_slot = ui.allocate_ui_with_layout(
        Vec2::new(safe_available_width(ui), editor_height),
        Layout::top_down(Align::Min),
        |ui| {
            Frame::new()
                .fill(fill)
                .stroke(Stroke::new(1.0, border_color()))
                .corner_radius(8)
                .inner_margin(Margin::symmetric(12, 10))
                .show(ui, |ui| {
                    let viewport_height = scroll_editor_viewport_height(editor_height);
                    ui.set_height(viewport_height);

                    ScrollArea::vertical()
                        .id_salt(scroll_id)
                        .max_height(viewport_height)
                        .min_scrolled_height(viewport_height)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.add_enabled_ui(enabled, |ui| ui.add(editor)).inner
                        })
                })
        },
    );

    let frame_output = editor_slot.inner;
    let editor_response = frame_output.inner.inner;

    if editor_response.has_focus() {
        ui.painter().rect_stroke(
            editor_slot.response.rect,
            CornerRadius::same(8),
            Stroke::new(1.5, accent_color()),
            egui::StrokeKind::Outside,
        );
    }

    editor_response
}

fn canvas_color() -> Color32 {
    Color32::from_rgb(241, 244, 247)
}

fn card_color() -> Color32 {
    Color32::from_rgb(255, 255, 255)
}

fn subtle_surface_color() -> Color32 {
    Color32::from_rgb(248, 250, 252)
}

fn tinted_surface_color() -> Color32 {
    Color32::from_rgb(246, 248, 250)
}

fn border_color() -> Color32 {
    Color32::from_rgb(210, 217, 225)
}

fn accent_color() -> Color32 {
    Color32::from_rgb(24, 78, 119)
}

fn text_primary_color() -> Color32 {
    Color32::from_rgb(32, 43, 56)
}

fn text_muted_color() -> Color32 {
    Color32::from_rgb(103, 115, 128)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;
    use std::time::Instant;

    use super::{
        TranslateDesktopApp, TranslationDirection, WorkerHandle, WorkerStatus, available_or,
        format_elapsed, panel_content_height, scroll_editor_viewport_height,
    };
    use crate::translate::TargetLang;

    #[test]
    fn maps_target_language() {
        assert_eq!(TranslationDirection::TrToEn.target_lang(), TargetLang::En);
        assert_eq!(TranslationDirection::EnToTr.target_lang(), TargetLang::Tr);
    }

    #[test]
    fn formats_sub_minute_elapsed_time() {
        assert_eq!(format_elapsed(Duration::from_secs(50)), "00:50");
    }

    #[test]
    fn formats_multi_minute_elapsed_time() {
        assert_eq!(format_elapsed(Duration::from_secs(200)), "3:20");
    }

    #[test]
    fn direction_change_keeps_source_text_and_clears_result() {
        let (command_tx, _command_rx) = mpsc::channel();
        let (_event_tx, event_rx) = mpsc::channel();

        let mut app = TranslateDesktopApp {
            source_text: "Hello world".into(),
            translated_text: "Merhaba dunya".into(),
            direction: TranslationDirection::EnToTr,
            is_translating: false,
            startup_started_at: Instant::now(),
            translation_started_at: None,
            status: WorkerStatus::Ready {
                device_label: "CPU".into(),
            },
            last_error: Some("old error".into()),
            startup_progress: None,
            worker: WorkerHandle {
                commands: command_tx,
                events: event_rx,
            },
        };

        app.set_direction(TranslationDirection::TrToEn);

        assert_eq!(app.source_text, "Hello world");
        assert!(app.translated_text.is_empty());
        assert!(app.last_error.is_none());
    }

    #[test]
    fn available_or_prefers_measured_height() {
        assert_eq!(available_or(640.0, 420.0), 640.0);
        assert_eq!(available_or(0.0, 240.0), 240.0);
    }

    #[test]
    fn panel_content_height_never_goes_negative() {
        assert_eq!(panel_content_height(640.0), 608.0);
        assert_eq!(panel_content_height(12.0), 0.0);
    }

    #[test]
    fn scroll_editor_viewport_height_accounts_for_frame_padding() {
        assert_eq!(scroll_editor_viewport_height(420.0), 398.0);
        assert_eq!(scroll_editor_viewport_height(18.0), 0.0);
    }
}
