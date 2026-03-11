use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use hf_hub::api::Progress;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Cache, Repo};
use ndarray::{Array2, Array3, ArrayView1, ArrayView3, Axis, Ix3};
use ort::ep::CPU;
#[cfg(target_os = "windows")]
use ort::ep::CUDA;
#[cfg(target_os = "macos")]
use ort::ep::CoreML;
#[cfg(target_os = "windows")]
use ort::ep::DirectML;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use ort::ep::ExecutionProvider;
#[cfg(target_os = "macos")]
use ort::ep::coreml::{
    ComputeUnits as CoreMLComputeUnits, ModelFormat as CoreMLModelFormat,
    SpecializationStrategy as CoreMLSpecializationStrategy,
};
#[cfg(target_os = "windows")]
use ort::ep::directml::PerformancePreference;
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::value::TensorRef;
use prost::Message;
use serde_json::Value;
use tokenizers::Tokenizer;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::normalizers::Precompiled;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence as PreTokenizerSequence;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const TR_EN_ONNX_REPO_ID: &str = "onnx-community/opus-mt-tc-big-tr-en";
const EN_TR_ONNX_REPO_ID: &str = "onnx-community/opus-mt-tc-big-en-tr";
const TR_EN_BASE_REPO_ID: &str = "Helsinki-NLP/opus-mt-tc-big-tr-en";
const EN_TR_BASE_REPO_ID: &str = "Helsinki-NLP/opus-mt-tc-big-en-tr";
#[cfg(target_os = "windows")]
const NVIDIA_VENDOR_ID: u32 = 0x10DE;
#[cfg(target_os = "windows")]
const AMD_VENDOR_ID: u32 = 0x1002;
#[cfg(target_os = "windows")]
const AMD_CPU_VENDOR_ID: u32 = 0x1022;
#[cfg(target_os = "windows")]
const INTEL_VENDOR_ID: u32 = 0x8086;
#[cfg(target_os = "windows")]
const MICROSOFT_VENDOR_ID: u32 = 0x1414;
#[cfg(target_os = "windows")]
const QUALCOMM_VENDOR_ID: u32 = 0x5143;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetLang {
    En,
    Tr,
}

struct ModelBundle {
    encoder: Session,
    decoder: Session,
    #[cfg(target_os = "macos")]
    encoder_path: PathBuf,
    #[cfg(target_os = "macos")]
    decoder_path: PathBuf,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    pad_token_id: i64,
    eos_token_id: i64,
}

impl ModelBundle {
    #[cfg(target_os = "macos")]
    fn build_sessions(&self, runtime_backend: RuntimeBackend) -> Result<(Session, Session)> {
        build_model_sessions(&self.encoder_path, &self.decoder_path, runtime_backend)
    }

    #[cfg(target_os = "macos")]
    fn replace_sessions(&mut self, encoder: Session, decoder: Session) {
        self.encoder = encoder;
        self.decoder = decoder;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecoderConfig {
    max_input_tokens: usize,
    max_new_tokens: usize,
    num_beams: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CoreMLSettings {
    compute_units: CoreMLComputeUnits,
    model_format: CoreMLModelFormat,
    static_input_shapes: bool,
    profile_compute_plan: bool,
    enable_cache: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeBackend {
    Cpu,
    #[cfg(target_os = "windows")]
    Cuda,
    #[cfg(target_os = "windows")]
    DirectML,
    #[cfg(target_os = "macos")]
    CoreML {
        settings: CoreMLSettings,
    },
}

impl RuntimeBackend {
    fn label(self) -> String {
        match self {
            Self::Cpu => "CPU".to_owned(),
            #[cfg(target_os = "windows")]
            Self::Cuda => detect_cuda_device_label(),
            #[cfg(target_os = "windows")]
            Self::DirectML => detect_directml_device_label(),
            #[cfg(target_os = "macos")]
            Self::CoreML { settings } => format_coreml_device_label(settings.compute_units),
        }
    }
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            max_input_tokens: 384,
            max_new_tokens: 300,
            num_beams: 4,
        }
    }
}

impl DecoderConfig {
    fn from_env() -> Self {
        Self {
            max_input_tokens: parse_env_usize(
                "TRANSLATE_MAX_INPUT_TOKENS",
                Self::default().max_input_tokens,
                96,
                448,
            ),
            max_new_tokens: parse_env_usize(
                "TRANSLATE_MAX_NEW_TOKENS",
                Self::default().max_new_tokens,
                64,
                512,
            ),
            num_beams: parse_env_usize("TRANSLATE_BEAMS", Self::default().num_beams, 1, 8),
        }
    }
}

struct ConfiguredRuntime {
    backend: RuntimeBackend,
    device_label: String,
}

impl ConfiguredRuntime {
    fn new(backend: RuntimeBackend) -> Self {
        Self {
            device_label: backend.label(),
            backend,
        }
    }
}

pub struct Translator {
    tr_en: ModelBundle,
    en_tr: ModelBundle,
    decoder_config: DecoderConfig,
    #[cfg(target_os = "macos")]
    runtime_backend: RuntimeBackend,
    device_label: String,
}

#[derive(Clone, Debug)]
pub struct StartupProgress {
    pub message: String,
    pub progress: f32,
}

impl Translator {
    pub fn new_with_progress<F>(mut on_progress: F) -> Result<Self>
    where
        F: FnMut(StartupProgress),
    {
        on_progress(StartupProgress {
            message: "Initializing ONNX Runtime.".to_owned(),
            progress: 0.03,
        });

        let configured_runtime =
            configure_ort().context("Failed to initialize ONNX Runtime execution providers")?;
        let decoder_config = DecoderConfig::from_env();

        on_progress(StartupProgress {
            message: "Preparing model cache.".to_owned(),
            progress: 0.08,
        });

        let default_cache = Cache::from_env();
        let api = ApiBuilder::from_cache(default_cache.clone())
            .build()
            .context("Failed to initialize Hugging Face Hub API client")?;
        let (runtime_backend, tr_en, en_tr) = load_model_bundles_with_runtime_fallback(
            &api,
            &default_cache,
            configured_runtime.backend,
            &mut on_progress,
        )?;
        let device_label = if runtime_backend == configured_runtime.backend {
            configured_runtime.device_label
        } else {
            runtime_backend.label()
        };

        Ok(Self {
            tr_en,
            en_tr,
            decoder_config,
            #[cfg(target_os = "macos")]
            runtime_backend,
            device_label,
        })
    }

    pub fn translate(&mut self, text: &str, target: TargetLang) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let layout = TranslationLayout::parse(text);
        if layout.translatable_segments.is_empty() {
            return Ok(text.to_owned());
        }

        let translated = match self.translate_layout(&layout, target) {
            Ok(translated) => translated,
            #[cfg(target_os = "macos")]
            Err(error) if self.should_fallback_to_cpu(&error) => {
                eprintln!("CoreML execution failed; rebuilding sessions on CPU: {error:#}");
                self.fallback_to_cpu().context(
                    "Failed to rebuild translation sessions on CPU after a CoreML runtime error",
                )?;
                self.translate_layout(&layout, target)
                    .context("CoreML execution failed, and the CPU retry also failed")?
            }
            Err(error) => return Err(error),
        };

        Ok(layout.rebuild(translated))
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    fn translate_layout(
        &mut self,
        layout: &TranslationLayout,
        target: TargetLang,
    ) -> Result<Vec<String>> {
        let model = match target {
            TargetLang::En => &mut self.tr_en,
            TargetLang::Tr => &mut self.en_tr,
        };

        layout
            .translatable_segments
            .iter()
            .map(|segment| translate_segment(model, self.decoder_config, segment))
            .collect()
    }

    #[cfg(target_os = "macos")]
    fn should_fallback_to_cpu(&self, error: &anyhow::Error) -> bool {
        matches!(self.runtime_backend, RuntimeBackend::CoreML { .. })
            && is_coreml_runtime_error(error)
    }

    #[cfg(target_os = "macos")]
    fn fallback_to_cpu(&mut self) -> Result<()> {
        if self.runtime_backend == RuntimeBackend::Cpu {
            return Ok(());
        }

        let (tr_en_encoder, tr_en_decoder) = self
            .tr_en
            .build_sessions(RuntimeBackend::Cpu)
            .context("Failed to rebuild Turkish -> English sessions on CPU")?;
        let (en_tr_encoder, en_tr_decoder) = self
            .en_tr
            .build_sessions(RuntimeBackend::Cpu)
            .context("Failed to rebuild English -> Turkish sessions on CPU")?;

        self.tr_en.replace_sessions(tr_en_encoder, tr_en_decoder);
        self.en_tr.replace_sessions(en_tr_encoder, en_tr_decoder);
        self.runtime_backend = RuntimeBackend::Cpu;
        self.device_label = RuntimeBackend::Cpu.label();

        Ok(())
    }
}

fn load_model_bundles<F>(
    api: &Api,
    cache: &Cache,
    runtime_backend: RuntimeBackend,
    on_progress: &mut F,
) -> Result<(ModelBundle, ModelBundle)>
where
    F: FnMut(StartupProgress),
{
    let mut tracker = ModelLoadProgressTracker::new(on_progress, 12);
    let tr_en = load_model_pair(
        api,
        cache,
        TR_EN_ONNX_REPO_ID,
        TR_EN_BASE_REPO_ID,
        runtime_backend,
        &mut tracker,
    )
    .with_context(|| format!("Failed loading model pair {TR_EN_ONNX_REPO_ID}"))?;
    let en_tr = load_model_pair(
        api,
        cache,
        EN_TR_ONNX_REPO_ID,
        EN_TR_BASE_REPO_ID,
        runtime_backend,
        &mut tracker,
    )
    .with_context(|| format!("Failed loading model pair {EN_TR_ONNX_REPO_ID}"))?;
    tracker.finish();
    Ok((tr_en, en_tr))
}

fn load_model_bundles_with_cache_fallback<F>(
    api: &Api,
    cache: &Cache,
    runtime_backend: RuntimeBackend,
    on_progress: &mut F,
) -> Result<(ModelBundle, ModelBundle)>
where
    F: FnMut(StartupProgress),
{
    match load_model_bundles(api, cache, runtime_backend, on_progress) {
        Ok(bundles) => Ok(bundles),
        Err(error) if is_file_exists_io_error(&error) => {
            let fallback_cache = fallback_hf_cache_dir();
            std::fs::create_dir_all(&fallback_cache).with_context(|| {
                format!(
                    "Failed to create fallback Hugging Face cache directory at {}",
                    fallback_cache.display()
                )
            })?;
            let fallback_api = ApiBuilder::new()
                .with_cache_dir(fallback_cache.clone())
                .build()
                .with_context(|| {
                    format!(
                        "Failed to initialize fallback Hugging Face API with cache {}",
                        fallback_cache.display()
                    )
                })?;
            let fallback_cache = Cache::new(fallback_cache.clone());
            let fallback_cache_path = fallback_cache.path().clone();
            load_model_bundles(&fallback_api, &fallback_cache, runtime_backend, on_progress)
                .with_context(|| {
                    format!(
                        "Retry with fallback Hugging Face cache failed at {}",
                        fallback_cache_path.display()
                    )
                })
        }
        Err(error) => Err(error),
    }
}

fn load_model_bundles_with_runtime_fallback<F>(
    api: &Api,
    cache: &Cache,
    preferred_backend: RuntimeBackend,
    on_progress: &mut F,
) -> Result<(RuntimeBackend, ModelBundle, ModelBundle)>
where
    F: FnMut(StartupProgress),
{
    let runtime_backends = runtime_backend_fallbacks(preferred_backend);
    let mut last_error = None;

    for (attempt, runtime_backend) in runtime_backends.into_iter().enumerate() {
        if attempt > 0 {
            on_progress(StartupProgress {
                message: format!(
                    "Retrying translation sessions on {}.",
                    runtime_backend.label()
                ),
                progress: 0.14,
            });
        }

        match load_model_bundles_with_cache_fallback(api, cache, runtime_backend, on_progress) {
            Ok((tr_en, en_tr)) => return Ok((runtime_backend, tr_en, en_tr)),
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error.expect("runtime backend fallback chain always tries at least one backend"))
}

fn is_file_exists_io_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let text = cause.to_string();
        text.contains("os error 17")
            || text.contains("File exists (os error 17)")
            || text.contains("I/O error File exists")
    })
}

fn fallback_hf_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/localtranslate/hf-hub");
    }

    std::env::temp_dir().join("localtranslate-hf-hub")
}

fn configure_ort() -> Result<ConfiguredRuntime> {
    ort::init().commit();

    let force_cpu = std::env::var("TRANSLATE_DEVICE")
        .map(|value| value.trim().eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);

    if force_cpu {
        return Ok(ConfiguredRuntime::new(RuntimeBackend::Cpu));
    }

    #[cfg(target_os = "windows")]
    {
        Ok(select_windows_runtime_backend())
    }

    #[cfg(target_os = "macos")]
    {
        let settings = coreml_settings_from_env();
        if probe_execution_provider(build_coreml_execution_provider(settings)).is_ok() {
            return Ok(ConfiguredRuntime::new(RuntimeBackend::CoreML { settings }));
        }

        return Ok(ConfiguredRuntime::new(RuntimeBackend::Cpu));
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    Ok(ConfiguredRuntime::new(RuntimeBackend::Cpu))
}

fn runtime_backend_fallbacks(preferred_backend: RuntimeBackend) -> Vec<RuntimeBackend> {
    let mut runtime_backends = vec![preferred_backend];

    #[cfg(target_os = "windows")]
    if preferred_backend == RuntimeBackend::Cuda {
        runtime_backends.push(RuntimeBackend::DirectML);
    }

    if preferred_backend != RuntimeBackend::Cpu {
        runtime_backends.push(RuntimeBackend::Cpu);
    }

    runtime_backends
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn probe_execution_provider<E>(provider: E) -> Result<()>
where
    E: ExecutionProvider,
{
    if !provider.supported_by_platform() {
        return Err(anyhow::anyhow!(
            "{} is not supported on this platform",
            provider.name()
        ));
    }

    let mut session_builder = Session::builder()?;

    provider
        .register(&mut session_builder)
        .map_err(anyhow::Error::from)
}

fn session_builder() -> Result<SessionBuilder> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_parallel_execution(true)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn build_model_sessions(
    encoder_path: &Path,
    decoder_path: &Path,
    runtime_backend: RuntimeBackend,
) -> Result<(Session, Session)> {
    let encoder = build_model_session(encoder_path, runtime_backend).with_context(|| {
        format!(
            "Failed to create encoder session from {}",
            encoder_path.display()
        )
    })?;
    let decoder = build_model_session(decoder_path, runtime_backend).with_context(|| {
        format!(
            "Failed to create decoder session from {}",
            decoder_path.display()
        )
    })?;
    Ok((encoder, decoder))
}

fn build_model_session(model_path: &Path, runtime_backend: RuntimeBackend) -> Result<Session> {
    #[cfg(target_os = "windows")]
    if runtime_backend == RuntimeBackend::Cuda {
        return session_builder()?
            .with_execution_providers([
                build_cuda_execution_provider().build().error_on_failure(),
                CPU::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(Into::into);
    }

    #[cfg(target_os = "windows")]
    if runtime_backend == RuntimeBackend::DirectML {
        return session_builder()?
            .with_execution_providers([
                build_directml_execution_provider()
                    .build()
                    .error_on_failure(),
                CPU::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(Into::into);
    }

    #[cfg(target_os = "macos")]
    if let RuntimeBackend::CoreML { settings } = runtime_backend {
        return session_builder()?
            .with_execution_providers([
                build_coreml_execution_provider(settings)
                    .build()
                    .error_on_failure(),
                CPU::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(Into::into);
    }

    session_builder()?
        .with_execution_providers([CPU::default().build()])
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(model_path)
        .map_err(Into::into)
}

#[cfg(target_os = "windows")]
fn detect_directml_device_label() -> String {
    preferred_windows_adapter_info()
        .map(|info| format_directml_device_label(&info))
        .unwrap_or_else(|| "DirectML (GPU)".to_owned())
}

#[cfg(target_os = "windows")]
fn detect_cuda_device_label() -> String {
    windows_adapter_info_for_vendor(NVIDIA_VENDOR_ID)
        .map(|info| format_cuda_device_label(&info))
        .unwrap_or_else(|| "CUDA (NVIDIA)".to_owned())
}

#[cfg(target_os = "windows")]
fn select_windows_runtime_backend() -> ConfiguredRuntime {
    let preferred_vendor = preferred_windows_adapter_info().map(|info| info.vendor);
    let cuda_probe = probe_execution_provider(build_cuda_execution_provider());
    let directml_probe = probe_execution_provider(build_directml_execution_provider());
    let cuda_usable = cuda_probe.is_ok();
    let directml_usable = directml_probe.is_ok();

    if should_prefer_cuda_for_vendor(preferred_vendor) && cuda_usable {
        ConfiguredRuntime::new(RuntimeBackend::Cuda)
    } else if directml_usable {
        if should_prefer_cuda_for_vendor(preferred_vendor) && !cuda_usable {
            eprintln!(
                "CUDA execution provider could not be initialized; falling back to DirectML. CUDA probe error: {}",
                cuda_probe
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "unknown error".to_owned())
            );
            ConfiguredRuntime {
                backend: RuntimeBackend::DirectML,
                device_label: annotate_device_label(
                    detect_directml_device_label(),
                    "CUDA unavailable",
                ),
            }
        } else {
            ConfiguredRuntime::new(RuntimeBackend::DirectML)
        }
    } else if cuda_usable {
        ConfiguredRuntime::new(RuntimeBackend::Cuda)
    } else {
        ConfiguredRuntime::new(RuntimeBackend::Cpu)
    }
}

#[cfg(target_os = "windows")]
fn annotate_device_label(label: String, note: &str) -> String {
    format!("{label} [{note}]")
}

#[cfg(target_os = "windows")]
fn should_prefer_cuda_for_vendor(vendor_id: Option<u32>) -> bool {
    vendor_id.is_none() || vendor_id == Some(NVIDIA_VENDOR_ID)
}

#[cfg(target_os = "windows")]
fn build_cuda_execution_provider() -> CUDA {
    CUDA::default()
}

#[cfg(target_os = "windows")]
fn build_directml_execution_provider() -> DirectML {
    DirectML::default().with_performance_preference(PerformancePreference::HighPerformance)
}

#[cfg(target_os = "windows")]
fn preferred_windows_adapter_info() -> Option<wgpu::AdapterInfo> {
    windows_adapter_infos()
        .into_iter()
        .max_by_key(score_adapter_for_directml)
}

#[cfg(target_os = "windows")]
fn windows_adapter_info_for_vendor(vendor_id: u32) -> Option<wgpu::AdapterInfo> {
    windows_adapter_infos()
        .into_iter()
        .filter(|info| info.vendor == vendor_id)
        .max_by_key(score_adapter_for_directml)
}

#[cfg(target_os = "windows")]
fn windows_adapter_infos() -> Vec<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..Default::default()
    });

    instance
        .enumerate_adapters(wgpu::Backends::DX12)
        .into_iter()
        .map(|adapter| adapter.get_info())
        .filter(|info| info.device_type != wgpu::DeviceType::Cpu)
        .collect()
}

#[cfg(target_os = "windows")]
fn score_adapter_for_directml(info: &wgpu::AdapterInfo) -> (u8, u8, u8) {
    let device_priority = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 5,
        wgpu::DeviceType::IntegratedGpu => 4,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 1,
    };
    let preferred_vendor = u8::from(!is_software_or_microsoft_adapter(info));
    let vendor_priority = match info.vendor {
        NVIDIA_VENDOR_ID => 4,
        AMD_VENDOR_ID | AMD_CPU_VENDOR_ID => 3,
        INTEL_VENDOR_ID => 2,
        QUALCOMM_VENDOR_ID => 1,
        _ => 0,
    };

    (device_priority, preferred_vendor, vendor_priority)
}

#[cfg(target_os = "windows")]
fn is_software_or_microsoft_adapter(info: &wgpu::AdapterInfo) -> bool {
    info.vendor == MICROSOFT_VENDOR_ID || info.name.contains("Microsoft")
}

#[cfg(target_os = "windows")]
fn format_directml_device_label(info: &wgpu::AdapterInfo) -> String {
    format_gpu_provider_label("DirectML", info)
}

#[cfg(target_os = "windows")]
fn format_cuda_device_label(info: &wgpu::AdapterInfo) -> String {
    format_gpu_provider_label("CUDA", info)
}

#[cfg(target_os = "windows")]
fn format_gpu_provider_label(provider_name: &str, info: &wgpu::AdapterInfo) -> String {
    let adapter_name = info.name.trim();
    if !adapter_name.is_empty() {
        return format!("{provider_name} ({adapter_name})");
    }

    let vendor_name = gpu_vendor_name(info.vendor);
    format!("{provider_name} ({vendor_name})")
}

#[cfg(target_os = "windows")]
fn gpu_vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        NVIDIA_VENDOR_ID => "NVIDIA",
        AMD_VENDOR_ID | AMD_CPU_VENDOR_ID => "AMD",
        INTEL_VENDOR_ID => "Intel",
        MICROSOFT_VENDOR_ID => "Microsoft",
        QUALCOMM_VENDOR_ID => "Qualcomm",
        _ => "GPU",
    }
}

#[cfg(target_os = "macos")]
fn build_coreml_execution_provider(settings: CoreMLSettings) -> CoreML {
    let mut coreml = CoreML::default()
        .with_compute_units(settings.compute_units)
        .with_model_format(settings.model_format)
        .with_static_input_shapes(settings.static_input_shapes)
        .with_profile_compute_plan(settings.profile_compute_plan)
        .with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction);
    if settings.enable_cache
        && let Some(cache_dir) = coreml_model_cache_dir(settings)
    {
        coreml = coreml.with_model_cache_dir(cache_dir.display().to_string());
    }
    coreml
}

#[cfg(target_os = "macos")]
fn coreml_settings_from_env() -> CoreMLSettings {
    CoreMLSettings {
        compute_units: coreml_compute_units_from_env(),
        model_format: coreml_model_format_from_env(),
        static_input_shapes: parse_env_bool("TRANSLATE_COREML_STATIC_INPUT_SHAPES", false),
        profile_compute_plan: parse_env_bool("TRANSLATE_COREML_PROFILE_PLAN", false),
        enable_cache: parse_env_bool("TRANSLATE_COREML_CACHE", true),
    }
}

#[cfg(target_os = "macos")]
fn coreml_compute_units_from_env() -> CoreMLComputeUnits {
    let requested = std::env::var("TRANSLATE_COREML_UNITS").ok();
    parse_coreml_compute_units(requested.as_deref())
}

#[cfg(target_os = "macos")]
fn coreml_model_format_from_env() -> CoreMLModelFormat {
    let requested = std::env::var("TRANSLATE_COREML_FORMAT").ok();
    parse_coreml_model_format(requested.as_deref())
}

#[cfg(target_os = "macos")]
fn parse_coreml_compute_units(raw: Option<&str>) -> CoreMLComputeUnits {
    match raw
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("ane")
        | Some("neural")
        | Some("neural-engine")
        | Some("neural_engine")
        | Some("cpuandneuralengine")
        | Some("cpu-neural-engine")
        | Some("cpu_and_neural_engine") => CoreMLComputeUnits::CPUAndNeuralEngine,
        Some("gpu") | Some("cpuandgpu") | Some("cpu-gpu") | Some("cpu_and_gpu") => {
            CoreMLComputeUnits::CPUAndGPU
        }
        Some("cpu") | Some("cpuonly") | Some("cpu-only") | Some("cpu_only") => {
            CoreMLComputeUnits::CPUOnly
        }
        _ => CoreMLComputeUnits::All,
    }
}

#[cfg(target_os = "macos")]
fn parse_coreml_model_format(raw: Option<&str>) -> CoreMLModelFormat {
    match raw
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mlprogram") | Some("ml_program") | Some("ml-program") => CoreMLModelFormat::MLProgram,
        _ => CoreMLModelFormat::NeuralNetwork,
    }
}

#[cfg(target_os = "macos")]
fn format_coreml_device_label(compute_units: CoreMLComputeUnits) -> String {
    match compute_units {
        CoreMLComputeUnits::All => "CoreML (Automatic)".to_owned(),
        CoreMLComputeUnits::CPUAndNeuralEngine => "CoreML (Neural Engine + CPU)".to_owned(),
        CoreMLComputeUnits::CPUAndGPU => "CoreML (GPU + CPU)".to_owned(),
        CoreMLComputeUnits::CPUOnly => "CoreML (CPU only)".to_owned(),
    }
}

#[cfg(target_os = "macos")]
fn coreml_model_cache_dir(settings: CoreMLSettings) -> Option<PathBuf> {
    let base = if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join("Library/Caches/LocalTranslateApp/coreml")
    } else {
        std::env::temp_dir().join("LocalTranslateApp/coreml")
    };
    let path = base.join(coreml_cache_namespace(settings));

    std::fs::create_dir_all(&path).ok()?;
    Some(path)
}

#[cfg(target_os = "macos")]
fn coreml_cache_namespace(settings: CoreMLSettings) -> String {
    format!(
        "units-{}_format-{}_shapes-{}",
        coreml_compute_units_slug(settings.compute_units),
        coreml_model_format_slug(settings.model_format),
        if settings.static_input_shapes {
            "static"
        } else {
            "dynamic"
        }
    )
}

#[cfg(target_os = "macos")]
fn coreml_compute_units_slug(compute_units: CoreMLComputeUnits) -> &'static str {
    match compute_units {
        CoreMLComputeUnits::All => "all",
        CoreMLComputeUnits::CPUAndNeuralEngine => "ane",
        CoreMLComputeUnits::CPUAndGPU => "gpu",
        CoreMLComputeUnits::CPUOnly => "cpu",
    }
}

#[cfg(target_os = "macos")]
fn coreml_model_format_slug(model_format: CoreMLModelFormat) -> &'static str {
    match model_format {
        CoreMLModelFormat::MLProgram => "mlprogram",
        CoreMLModelFormat::NeuralNetwork => "neuralnetwork",
    }
}

#[cfg(target_os = "macos")]
fn is_coreml_runtime_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("CoreMLExecutionProvider")
            || message.contains("Unable to compute the prediction using a neural network model")
            || (message.contains("CoreML") && message.contains("error code: -1"))
    })
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "no" | "off" => false,
            "1" | "true" | "yes" | "on" => true,
            _ => default,
        })
        .unwrap_or(default)
}

fn coreml_debug_enabled() -> bool {
    parse_env_bool("TRANSLATE_COREML_DEBUG", false)
}

fn parse_env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn summarize_i64_tensor(name: &str, values: &[i64], shape: &[usize]) -> String {
    let min = values.iter().copied().min().unwrap_or_default();
    let max = values.iter().copied().max().unwrap_or_default();
    format!("{name}=i64{:?} min={min} max={max}", shape)
}

fn summarize_f32_shape(name: &str, shape: &[usize]) -> String {
    format!("{name}=f32{:?}", shape)
}

#[cfg(target_os = "macos")]
fn log_session_io(label: &str, session: &Session) {
    eprintln!("CoreML debug: session `{label}` inputs:");
    for input in session.inputs() {
        eprintln!("  - {}: {}", input.name(), input.dtype());
    }
    eprintln!("CoreML debug: session `{label}` outputs:");
    for output in session.outputs() {
        eprintln!("  - {}: {}", output.name(), output.dtype());
    }
}

fn load_model_pair<F>(
    api: &Api,
    cache: &Cache,
    onnx_repo_id: &str,
    base_repo_id: &str,
    runtime_backend: RuntimeBackend,
    tracker: &mut ModelLoadProgressTracker<'_, F>,
) -> Result<ModelBundle>
where
    F: FnMut(StartupProgress),
{
    let encoder_path =
        resolve_model_file(api, cache, onnx_repo_id, "onnx/encoder_model.onnx", tracker)
            .with_context(|| format!("Could not fetch encoder ONNX file from {onnx_repo_id}"))?;
    let decoder_path =
        resolve_model_file(api, cache, onnx_repo_id, "onnx/decoder_model.onnx", tracker)
            .with_context(|| format!("Could not fetch decoder ONNX file from {onnx_repo_id}"))?;
    let tokenizer_json_path =
        resolve_model_file(api, cache, onnx_repo_id, "tokenizer.json", tracker)
            .with_context(|| format!("Could not fetch tokenizer.json from {onnx_repo_id}"))?;
    let source_spm_path = resolve_model_file(api, cache, base_repo_id, "source.spm", tracker)
        .with_context(|| format!("Could not fetch source.spm from {base_repo_id}"))?;
    let target_spm_path = resolve_model_file(api, cache, base_repo_id, "target.spm", tracker)
        .with_context(|| format!("Could not fetch target.spm from {base_repo_id}"))?;
    let vocab_path = resolve_model_file(api, cache, base_repo_id, "vocab.json", tracker)
        .with_context(|| format!("Could not fetch vocab.json from {base_repo_id}"))?;

    let (encoder, decoder) = build_model_sessions(&encoder_path, &decoder_path, runtime_backend)
        .with_context(|| {
            format!(
                "Could not create ONNX sessions for {onnx_repo_id} with backend {}",
                runtime_backend.label()
            )
        })?;
    #[cfg(target_os = "macos")]
    if matches!(runtime_backend, RuntimeBackend::CoreML { .. }) && coreml_debug_enabled() {
        log_session_io(&format!("{onnx_repo_id} encoder"), &encoder);
        log_session_io(&format!("{onnx_repo_id} decoder"), &decoder);
    }
    let source_tokenizer =
        load_sentencepiece_tokenizer(&source_spm_path, &vocab_path, base_repo_id)
            .with_context(|| format!("Source tokenizer could not be loaded ({base_repo_id})"))?;
    let target_tokenizer =
        load_sentencepiece_tokenizer(&target_spm_path, &vocab_path, base_repo_id)
            .with_context(|| format!("Target tokenizer could not be loaded ({base_repo_id})"))?;
    let (pad_token_id, eos_token_id) = load_special_token_ids(&tokenizer_json_path, onnx_repo_id)?;

    Ok(ModelBundle {
        encoder,
        decoder,
        #[cfg(target_os = "macos")]
        encoder_path,
        #[cfg(target_os = "macos")]
        decoder_path,
        source_tokenizer,
        target_tokenizer,
        pad_token_id,
        eos_token_id,
    })
}

fn resolve_model_file<F>(
    api: &Api,
    cache: &Cache,
    repo_id: &str,
    filename: &str,
    tracker: &mut ModelLoadProgressTracker<'_, F>,
) -> Result<PathBuf>
where
    F: FnMut(StartupProgress),
{
    let repo = Repo::model(repo_id.to_owned());
    if let Some(path) = cache.repo(repo.clone()).get(filename) {
        tracker.mark_cached(repo_id, filename);
        return Ok(path);
    }

    tracker.start_download(repo_id, filename);
    let progress = HubDownloadProgress::new(tracker, repo_id, filename);
    let downloaded = api
        .repo(repo)
        .download_with_progress(filename, progress)
        .with_context(|| format!("Download failed for {repo_id}/{filename}"))?;
    Ok(downloaded)
}

struct ModelLoadProgressTracker<'a, F: FnMut(StartupProgress)> {
    on_progress: &'a mut F,
    total_files: usize,
    completed_files: usize,
}

impl<'a, F: FnMut(StartupProgress)> ModelLoadProgressTracker<'a, F> {
    fn new(on_progress: &'a mut F, total_files: usize) -> Self {
        let mut tracker = Self {
            on_progress,
            total_files,
            completed_files: 0,
        };
        tracker.emit(
            "Checking local model files.".to_owned(),
            tracker.progress_value(0.0),
        );
        tracker
    }

    fn mark_cached(&mut self, repo_id: &str, filename: &str) {
        self.completed_files += 1;
        self.emit(
            format!("Using cached asset: {repo_id}/{filename}"),
            self.progress_value(0.0),
        );
    }

    fn start_download(&mut self, repo_id: &str, filename: &str) {
        self.emit(
            format!("Downloading model asset: {repo_id}/{filename}"),
            self.progress_value(0.0),
        );
    }

    fn update_download(&mut self, repo_id: &str, filename: &str, fraction_in_file: f32) {
        let percent = (fraction_in_file * 100.0).round() as u8;
        self.emit(
            format!("Downloading {repo_id}/{filename} ({percent}%)"),
            self.progress_value(fraction_in_file),
        );
    }

    fn finish_download(&mut self, repo_id: &str, filename: &str) {
        self.completed_files += 1;
        self.emit(
            format!("Model asset ready: {repo_id}/{filename}"),
            self.progress_value(0.0),
        );
    }

    fn finish(&mut self) {
        self.emit("Model runtime ready.".to_owned(), 1.0);
    }

    fn progress_value(&self, file_fraction: f32) -> f32 {
        let startup_floor = 0.08;
        let startup_ceiling = 0.96;
        let span = startup_ceiling - startup_floor;
        let done = self.completed_files as f32 + file_fraction.clamp(0.0, 1.0);
        let raw = done / self.total_files as f32;
        (startup_floor + span * raw).clamp(0.0, startup_ceiling)
    }

    fn emit(&mut self, message: String, progress: f32) {
        (self.on_progress)(StartupProgress { message, progress });
    }
}

struct HubDownloadProgress<'a, 'b, F: FnMut(StartupProgress)> {
    tracker: &'a mut ModelLoadProgressTracker<'b, F>,
    repo_id: String,
    filename: String,
    total_size: usize,
    downloaded: usize,
    last_emit: Option<Instant>,
}

impl<'a, 'b, F: FnMut(StartupProgress)> HubDownloadProgress<'a, 'b, F> {
    fn new(
        tracker: &'a mut ModelLoadProgressTracker<'b, F>,
        repo_id: &str,
        filename: &str,
    ) -> Self {
        Self {
            tracker,
            repo_id: repo_id.to_owned(),
            filename: filename.to_owned(),
            total_size: 0,
            downloaded: 0,
            last_emit: None,
        }
    }

    fn file_fraction(&self) -> f32 {
        if self.total_size == 0 {
            0.0
        } else {
            (self.downloaded as f32 / self.total_size as f32).clamp(0.0, 1.0)
        }
    }
}

impl<F: FnMut(StartupProgress)> Progress for HubDownloadProgress<'_, '_, F> {
    fn init(&mut self, size: usize, _filename: &str) {
        self.total_size = size.max(1);
        self.downloaded = 0;
        self.last_emit = None;
        self.tracker
            .update_download(&self.repo_id, &self.filename, self.file_fraction());
    }

    fn update(&mut self, size: usize) {
        self.downloaded = self.downloaded.saturating_add(size).min(self.total_size);
        let now = Instant::now();
        let should_emit = self
            .last_emit
            .map(|last| now.duration_since(last) >= Duration::from_millis(90))
            .unwrap_or(true);
        if should_emit {
            self.last_emit = Some(now);
            self.tracker
                .update_download(&self.repo_id, &self.filename, self.file_fraction());
        }
    }

    fn finish(&mut self) {
        self.downloaded = self.total_size;
        self.tracker.finish_download(&self.repo_id, &self.filename);
    }
}

fn load_sentencepiece_tokenizer(
    tokenizer_path: &Path,
    vocab_path: &Path,
    repo_id: &str,
) -> Result<Tokenizer> {
    let bytes = std::fs::read(tokenizer_path)
        .with_context(|| format!("SentencePiece model could not be read ({repo_id})"))?;
    let model = SentencePieceModelProto::decode(bytes.as_slice())
        .with_context(|| format!("SentencePiece model could not be parsed ({repo_id})"))?;
    let vocab_json = std::fs::read_to_string(vocab_path)
        .with_context(|| format!("Vocab file could not be read ({repo_id})"))?;
    let vocab_map: Value = serde_json::from_str(&vocab_json)
        .with_context(|| format!("Vocab JSON could not be parsed ({repo_id})"))?;

    let normalizer_spec = model
        .normalizer_spec
        .with_context(|| format!("SentencePiece normalizer_spec missing ({repo_id})"))?;
    let trainer_spec = model.trainer_spec.unwrap_or_default();

    let piece_scores = model
        .pieces
        .into_iter()
        .map(|piece| {
            (
                piece.piece.unwrap_or_default(),
                f64::from(piece.score.unwrap_or_default()),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();

    let vocab_object = vocab_map
        .as_object()
        .with_context(|| format!("Vocab JSON must be an object ({repo_id})"))?;
    let max_id = vocab_object
        .values()
        .filter_map(Value::as_u64)
        .max()
        .with_context(|| format!("Vocab JSON is empty ({repo_id})"))? as usize;
    let mut vocab = vec![(String::new(), 0.0); max_id + 1];

    for (token, id_value) in vocab_object {
        let id = id_value
            .as_u64()
            .with_context(|| format!("Vocab id is not numeric for token `{token}` ({repo_id})"))?
            as usize;
        let score = piece_scores.get(token).copied().unwrap_or(0.0);
        vocab[id] = (token.clone(), score);
    }

    let unk_id = vocab_object
        .get("<unk>")
        .and_then(Value::as_u64)
        .with_context(|| format!("Vocab token id missing for <unk> ({repo_id})"))?
        as usize;
    let mut tokenizer = Tokenizer::new(
        Unigram::from(
            vocab,
            Some(unk_id),
            trainer_spec.byte_fallback.unwrap_or(false),
        )
        .map_err(|e| anyhow::anyhow!("Tokenizer model build failed: {e}"))?,
    );

    let normalizer = Precompiled::from(
        normalizer_spec
            .precompiled_charsmap
            .as_deref()
            .with_context(|| format!("SentencePiece precompiled_charsmap missing ({repo_id})"))?,
    )
    .with_context(|| format!("SentencePiece precompiled_charsmap invalid ({repo_id})"))?;
    let metaspace = Metaspace::new('▁', PrependScheme::Always, true);

    tokenizer.with_normalizer(Some(NormalizerWrapper::from(normalizer)));
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::from(PreTokenizerSequence::new(
        vec![
            PreTokenizerWrapper::from(WhitespaceSplit),
            PreTokenizerWrapper::from(metaspace.clone()),
        ],
    ))));
    tokenizer.with_decoder(Some(DecoderWrapper::from(metaspace)));

    Ok(tokenizer)
}

fn load_special_token_ids(tokenizer_path: &Path, repo_id: &str) -> Result<(i64, i64)> {
    let json_text = std::fs::read_to_string(tokenizer_path)
        .with_context(|| format!("Tokenizer file could not be read ({repo_id})"))?;
    let json: Value = serde_json::from_str(&json_text)
        .with_context(|| format!("Tokenizer JSON could not be parsed ({repo_id})"))?;

    let pad_token_id = added_token_id(&json, "<pad>", repo_id)?;
    let eos_token_id = added_token_id(&json, "</s>", repo_id)?;

    Ok((pad_token_id, eos_token_id))
}

fn added_token_id(json: &Value, token: &str, repo_id: &str) -> Result<i64> {
    let added_tokens = json
        .get("added_tokens")
        .and_then(Value::as_array)
        .with_context(|| format!("Tokenizer added_tokens missing ({repo_id})"))?;

    added_tokens
        .iter()
        .find(|entry| entry.get("content").and_then(Value::as_str) == Some(token))
        .and_then(|entry| entry.get("id").and_then(Value::as_i64))
        .with_context(|| format!("Tokenizer token id missing for `{token}` ({repo_id})"))
}

fn translate_segment(
    model: &mut ModelBundle,
    decoder_config: DecoderConfig,
    text: &str,
) -> Result<String> {
    let token_count = encode_source_ids(&model.source_tokenizer, model.eos_token_id, text)?.len();

    if token_count <= decoder_config.max_input_tokens {
        return translate_chunk(model, decoder_config, text);
    }

    let sentences = split_sentences(text);

    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();

    for sentence in &sentences {
        let candidate = if current_chunk.is_empty() {
            sentence.clone()
        } else {
            format!("{current_chunk}{sentence}")
        };

        let count = encode_source_ids(
            &model.source_tokenizer,
            model.eos_token_id,
            candidate.as_str(),
        )?
        .len();

        if count <= decoder_config.max_input_tokens {
            current_chunk = candidate;
        } else {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
            }
            current_chunk = sentence.clone();
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    let mut results = Vec::new();
    for chunk in &chunks {
        results.push(translate_chunk(model, decoder_config, chunk.trim())?);
    }

    Ok(results.join(" "))
}

/// Metni cümle sınırlarından (`. `, `? `, `! `) böler.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut remaining = text;

    loop {
        let boundary = [
            remaining.find(". "),
            remaining.find("? "),
            remaining.find("! "),
        ]
        .into_iter()
        .flatten()
        .min();

        match boundary {
            Some(pos) => {
                sentences.push(remaining[..pos + 2].to_owned());
                remaining = &remaining[pos + 2..];
            }
            None => {
                if !remaining.is_empty() {
                    sentences.push(remaining.to_owned());
                }
                break;
            }
        }
    }

    sentences
}

fn translate_chunk(
    model: &mut ModelBundle,
    decoder_config: DecoderConfig,
    text: &str,
) -> Result<String> {
    let src_ids = encode_source_ids(&model.source_tokenizer, model.eos_token_id, text)?;
    let src_mask: Vec<i64> = vec![1; src_ids.len()];
    if coreml_debug_enabled() {
        eprintln!(
            "CoreML debug: translate_chunk chars={} src_tokens={} beams={} max_new_tokens={}",
            text.chars().count(),
            src_ids.len(),
            decoder_config.num_beams,
            decoder_config.max_new_tokens
        );
    }
    let encoder_hidden = run_encoder(&mut model.encoder, &src_ids, &src_mask)?;

    if decoder_config.num_beams <= 1 {
        return greedy_decode_chunk(
            &mut model.decoder,
            &model.target_tokenizer,
            model.pad_token_id,
            model.eos_token_id,
            decoder_config.max_new_tokens,
            &encoder_hidden,
            &src_mask,
        );
    }

    beam_search_decode_chunk(
        &mut model.decoder,
        &model.target_tokenizer,
        model.pad_token_id,
        model.eos_token_id,
        decoder_config,
        &encoder_hidden,
        &src_mask,
    )
}

fn run_encoder(encoder: &mut Session, src_ids: &[i64], src_mask: &[i64]) -> Result<Array3<f32>> {
    let ids = Array2::from_shape_vec((1, src_ids.len()), src_ids.to_vec())?;
    let mask = Array2::from_shape_vec((1, src_mask.len()), src_mask.to_vec())?;
    let ids_summary = summarize_i64_tensor("input_ids", src_ids, &[1, src_ids.len()]);
    let mask_summary = summarize_i64_tensor("attention_mask", src_mask, &[1, src_mask.len()]);
    if coreml_debug_enabled() {
        eprintln!("CoreML debug: encoder inputs: {ids_summary}; {mask_summary}");
    }
    let ids_t = TensorRef::from_array_view(ids.view()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mask_t = TensorRef::from_array_view(mask.view()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let out = encoder
        .run(ort::inputs![
            "input_ids" => ids_t,
            "attention_mask" => mask_t
        ])
        .with_context(|| format!("Encoder inference failed with {ids_summary}; {mask_summary}"))?;

    Ok(out["last_hidden_state"]
        .try_extract_array::<f32>()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .into_dimensionality::<Ix3>()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .to_owned())
}

fn greedy_decode_chunk(
    decoder: &mut Session,
    target_tokenizer: &Tokenizer,
    pad_token_id: i64,
    eos_token_id: i64,
    max_new_tokens: usize,
    encoder_hidden: &Array3<f32>,
    src_mask: &[i64],
) -> Result<String> {
    let mut generated = vec![pad_token_id];

    for _ in 0..max_new_tokens {
        let target = Array2::from_shape_vec((1, generated.len()), generated.clone())?;
        let next_token =
            with_decoder_logits(decoder, encoder_hidden, src_mask, &target, |logits| {
                let beam_logits = logits.index_axis(Axis(0), 0);
                let last_logits = beam_logits.index_axis(Axis(0), generated.len() - 1);
                best_token_id(last_logits, pad_token_id as usize).map(|token| token as i64)
            })?;

        if next_token == eos_token_id {
            break;
        }

        generated.push(next_token);
    }

    let output_ids: Vec<u32> = generated
        .into_iter()
        .skip(1)
        .filter(|&token| token != eos_token_id)
        .map(|token| token as u32)
        .collect();

    decode_target_ids(target_tokenizer, pad_token_id, eos_token_id, &output_ids)
}

fn beam_search_decode_chunk(
    decoder: &mut Session,
    target_tokenizer: &Tokenizer,
    pad_token_id: i64,
    eos_token_id: i64,
    decoder_config: DecoderConfig,
    encoder_hidden: &Array3<f32>,
    src_mask: &[i64],
) -> Result<String> {
    let mut beams: Vec<(Vec<i64>, f32)> = vec![(vec![pad_token_id], 0.0)];
    let mut completed: Vec<(Vec<i64>, f32)> = Vec::new();

    for _ in 0..decoder_config.max_new_tokens {
        if beams.is_empty() {
            break;
        }

        let target = build_beam_input(&beams)?;
        let target_len = target.ncols();
        let mut candidates: Vec<(Vec<i64>, f32)> = Vec::new();

        with_decoder_logits(decoder, encoder_hidden, src_mask, &target, |logits| {
            for (beam_index, (beam_ids, beam_score)) in beams.iter().enumerate() {
                let beam_logits = logits.index_axis(Axis(0), beam_index);
                let last_logits = beam_logits.index_axis(Axis(0), target_len - 1);

                for (token_idx, log_prob) in top_log_prob_tokens(
                    last_logits,
                    pad_token_id as usize,
                    2 * decoder_config.num_beams,
                ) {
                    let mut new_ids = beam_ids.clone();
                    new_ids.push(token_idx as i64);
                    let new_score = beam_score + log_prob;

                    if token_idx as i64 == eos_token_id {
                        completed.push((new_ids, new_score));
                    } else {
                        candidates.push((new_ids, new_score));
                    }
                }
            }

            Ok(())
        })?;

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        beams = candidates
            .into_iter()
            .take(decoder_config.num_beams)
            .collect();

        if completed.len() >= decoder_config.num_beams && !beams.is_empty() {
            completed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let best_incomplete = beams[0].1;
            let worst_complete = completed[decoder_config.num_beams - 1].1;
            if best_incomplete < worst_complete {
                break;
            }
        }
    }

    completed.extend(beams);

    let best = completed
        .into_iter()
        .max_by(|a, b| {
            let sa = a.1 / (a.0.len() as f32);
            let sb = b.1 / (b.0.len() as f32);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .context("No translation beam produced an output")?;

    let output_ids: Vec<u32> = best
        .0
        .into_iter()
        .skip(1)
        .filter(|&token| token != eos_token_id)
        .map(|token| token as u32)
        .collect();

    decode_target_ids(target_tokenizer, pad_token_id, eos_token_id, &output_ids)
}

fn build_beam_input(beams: &[(Vec<i64>, f32)]) -> Result<Array2<i64>> {
    let beam_count = beams.len();
    let target_len = beams
        .first()
        .map(|(tokens, _)| tokens.len())
        .context("Beam input cannot be empty")?;
    let flat_input: Vec<i64> = beams
        .iter()
        .flat_map(|(tokens, _)| tokens.iter().copied())
        .collect();

    Array2::from_shape_vec((beam_count, target_len), flat_input).map_err(Into::into)
}

fn with_decoder_logits<T, F>(
    decoder: &mut Session,
    encoder_hidden: &Array3<f32>,
    src_mask: &[i64],
    target: &Array2<i64>,
    decode_step: F,
) -> Result<T>
where
    F: FnOnce(ArrayView3<'_, f32>) -> Result<T>,
{
    let beam_count = target.nrows();
    let encoder_attention_mask =
        Array2::from_shape_vec((beam_count, src_mask.len()), src_mask.repeat(beam_count))?;
    let expanded_hidden = if beam_count == 1 {
        None
    } else {
        Some(expand_encoder_hidden(encoder_hidden, beam_count)?)
    };
    let hidden_view = expanded_hidden
        .as_ref()
        .map(|hidden| hidden.view())
        .unwrap_or_else(|| encoder_hidden.view());

    let target_t = TensorRef::from_array_view(target.view()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let hidden_t = TensorRef::from_array_view(hidden_view).map_err(|e| anyhow::anyhow!("{e}"))?;
    let enc_mask_t = TensorRef::from_array_view(encoder_attention_mask.view())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let target_summary = summarize_i64_tensor(
        "input_ids",
        target.as_slice().unwrap_or(&[]),
        &[beam_count, target.ncols()],
    );
    let hidden_summary = summarize_f32_shape(
        "encoder_hidden_states",
        &[
            hidden_view.dim().0,
            hidden_view.dim().1,
            hidden_view.dim().2,
        ],
    );
    let enc_mask_summary = summarize_i64_tensor(
        "encoder_attention_mask",
        encoder_attention_mask.as_slice().unwrap_or(&[]),
        &[beam_count, src_mask.len()],
    );
    if coreml_debug_enabled() {
        eprintln!(
            "CoreML debug: decoder inputs: beam_count={} target_len={}; {}; {}; {}",
            beam_count,
            target.ncols(),
            target_summary,
            hidden_summary,
            enc_mask_summary
        );
    }

    let out = decoder
        .run(ort::inputs![
            "input_ids" => target_t,
            "encoder_hidden_states" => hidden_t,
            "encoder_attention_mask" => enc_mask_t
        ])
        .with_context(|| {
            format!(
                "Decoder inference failed at beam_count={} target_len={} with {}; {}; {}",
                beam_count,
                target.ncols(),
                target_summary,
                hidden_summary,
                enc_mask_summary
            )
        })?;
    let logits = out["logits"]
        .try_extract_array::<f32>()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .into_dimensionality::<Ix3>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    decode_step(logits)
}

fn expand_encoder_hidden(encoder_hidden: &Array3<f32>, batch_size: usize) -> Result<Array3<f32>> {
    let (_, src_len, hidden_size) = encoder_hidden.dim();
    Ok(encoder_hidden
        .view()
        .broadcast((batch_size, src_len, hidden_size))
        .context("Encoder hidden state could not be broadcast for batched decode")?
        .to_owned())
}

fn best_token_id(logits: ArrayView1<'_, f32>, blocked_token_id: usize) -> Result<usize> {
    logits
        .iter()
        .enumerate()
        .filter(|(index, value)| *index != blocked_token_id && value.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .context("Decoder did not return a valid token")
}

fn top_log_prob_tokens(
    logits: ArrayView1<'_, f32>,
    blocked_token_id: usize,
    count: usize,
) -> Vec<(usize, f32)> {
    let mut raw: Vec<f32> = logits.iter().copied().collect();
    if let Some(value) = raw.get_mut(blocked_token_id) {
        *value = f32::NEG_INFINITY;
    }

    let max_val = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = raw
        .iter()
        .filter(|value| value.is_finite())
        .map(|&value| (value - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;
    let mut indexed: Vec<(usize, f32)> = raw
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (index, value - log_sum_exp))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(count);
    indexed
}

fn encode_source_ids(tokenizer: &Tokenizer, eos_token_id: i64, text: &str) -> Result<Vec<i64>> {
    let mut ids: Vec<i64> = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("Encode failed: {e}"))?
        .get_ids()
        .iter()
        .map(|&piece| i64::from(piece))
        .collect();
    ids.push(eos_token_id);
    Ok(ids)
}

fn decode_target_ids(
    tokenizer: &Tokenizer,
    pad_token_id: i64,
    eos_token_id: i64,
    token_ids: &[u32],
) -> Result<String> {
    let filtered: Vec<u32> = token_ids
        .iter()
        .copied()
        .filter(|&id| id != pad_token_id as u32 && id != eos_token_id as u32)
        .collect();

    tokenizer
        .decode(&filtered, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))
}

#[derive(Clone, PartialEq, Message)]
struct SentencePieceModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SentencePieceProto>,
    #[prost(message, optional, tag = "2")]
    trainer_spec: Option<TrainerSpecProto>,
    #[prost(message, optional, tag = "3")]
    normalizer_spec: Option<NormalizerSpecProto>,
}

#[derive(Clone, PartialEq, Message)]
struct SentencePieceProto {
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    #[prost(float, optional, tag = "2")]
    score: Option<f32>,
}

#[derive(Clone, PartialEq, Message)]
struct TrainerSpecProto {
    #[prost(bool, optional, tag = "35")]
    byte_fallback: Option<bool>,
    #[prost(int32, optional, tag = "40")]
    unk_id: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct NormalizerSpecProto {
    #[prost(bytes = "vec", optional, tag = "2")]
    precompiled_charsmap: Option<Vec<u8>>,
}

#[derive(Debug, Eq, PartialEq)]
struct TranslationLayout {
    segments: Vec<LayoutSegment>,
    translatable_segments: Vec<String>,
}

impl TranslationLayout {
    fn parse(text: &str) -> Self {
        let mut segments = Vec::new();
        let mut translatable_segments = Vec::new();

        for line in text.split_inclusive('\n') {
            let (content, newline) = match line.strip_suffix('\n') {
                Some(content) => (content, "\n"),
                None => (line, ""),
            };

            let trimmed_start = content.trim_start_matches(char::is_whitespace);
            let trimmed = trimmed_start.trim_end_matches(char::is_whitespace);

            if trimmed.is_empty() {
                segments.push(LayoutSegment::Whitespace {
                    text: content.to_owned(),
                    newline: newline.to_owned(),
                });
                continue;
            }

            let leading_len = content.len() - trimmed_start.len();
            let trailing_start = leading_len + trimmed.len();

            segments.push(LayoutSegment::Translatable {
                leading: content[..leading_len].to_owned(),
                translated_index: translatable_segments.len(),
                trailing: content[trailing_start..].to_owned(),
                newline: newline.to_owned(),
            });
            translatable_segments.push(trimmed.to_owned());
        }

        Self {
            segments,
            translatable_segments,
        }
    }

    fn rebuild(self, translated_segments: Vec<String>) -> String {
        let mut output = String::new();

        for segment in self.segments {
            match segment {
                LayoutSegment::Whitespace { text, newline } => {
                    output.push_str(&text);
                    output.push_str(&newline);
                }
                LayoutSegment::Translatable {
                    leading,
                    translated_index,
                    trailing,
                    newline,
                } => {
                    output.push_str(&leading);
                    output.push_str(
                        translated_segments
                            .get(translated_index)
                            .map(String::as_str)
                            .unwrap_or_default(),
                    );
                    output.push_str(&trailing);
                    output.push_str(&newline);
                }
            }
        }

        output
    }
}

#[derive(Debug, Eq, PartialEq)]
enum LayoutSegment {
    Whitespace {
        text: String,
        newline: String,
    },
    Translatable {
        leading: String,
        translated_index: usize,
        trailing: String,
        newline: String,
    },
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "windows")]
    use super::{
        AMD_VENDOR_ID, MICROSOFT_VENDOR_ID, NVIDIA_VENDOR_ID, annotate_device_label,
        format_cuda_device_label, format_directml_device_label, score_adapter_for_directml,
        should_prefer_cuda_for_vendor,
    };
    use super::{DecoderConfig, TranslationLayout, parse_env_bool, parse_env_usize};
    #[cfg(target_os = "macos")]
    use super::{
        format_coreml_device_label, parse_coreml_compute_units, parse_coreml_model_format,
    };
    #[cfg(target_os = "macos")]
    use ort::ep::coreml::{ComputeUnits as CoreMLComputeUnits, ModelFormat as CoreMLModelFormat};

    #[cfg(target_os = "windows")]
    fn adapter_info(name: &str, vendor: u32, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_owned(),
            vendor,
            device: 0,
            device_type,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Dx12,
        }
    }

    #[test]
    fn preserves_line_breaks_and_outer_whitespace() {
        let layout = TranslationLayout::parse("  merhaba  \n\n dunya \n");

        assert_eq!(
            layout.rebuild(vec!["hello".into(), "world".into()]),
            "  hello  \n\n world \n"
        );
    }

    #[test]
    fn returns_original_text_for_whitespace_only_input() {
        let layout = TranslationLayout::parse(" \n\t\n");
        assert!(layout.translatable_segments.is_empty());
        assert_eq!(layout.rebuild(Vec::new()), " \n\t\n");
    }

    #[test]
    fn env_parser_uses_default_when_value_missing() {
        assert_eq!(parse_env_usize("MISSING_ENV", 12, 4, 24), 12);
        assert_eq!(DecoderConfig::default().num_beams, 4);
        assert!(parse_env_bool("MISSING_BOOL_ENV", true));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn coreml_compute_units_parser_supports_shortcuts() {
        assert_eq!(
            parse_coreml_compute_units(Some("ane")),
            CoreMLComputeUnits::CPUAndNeuralEngine
        );
        assert_eq!(
            parse_coreml_compute_units(Some("gpu")),
            CoreMLComputeUnits::CPUAndGPU
        );
        assert_eq!(
            parse_coreml_compute_units(Some("cpu")),
            CoreMLComputeUnits::CPUOnly
        );
        assert_eq!(
            parse_coreml_compute_units(Some("bogus")),
            CoreMLComputeUnits::All
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn coreml_device_label_reflects_selected_compute_units() {
        assert_eq!(
            format_coreml_device_label(CoreMLComputeUnits::CPUAndNeuralEngine),
            "CoreML (Neural Engine + CPU)"
        );
        assert_eq!(
            format_coreml_device_label(CoreMLComputeUnits::CPUOnly),
            "CoreML (CPU only)"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn coreml_model_format_parser_supports_mlprogram() {
        assert_eq!(
            parse_coreml_model_format(Some("mlprogram")),
            CoreMLModelFormat::MLProgram
        );
        assert_eq!(
            parse_coreml_model_format(Some("ml-program")),
            CoreMLModelFormat::MLProgram
        );
        assert_eq!(
            parse_coreml_model_format(Some("bogus")),
            CoreMLModelFormat::NeuralNetwork
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn uses_adapter_name_for_directml_label() {
        let info = adapter_info(
            "NVIDIA GeForce RTX 4070 SUPER",
            NVIDIA_VENDOR_ID,
            wgpu::DeviceType::DiscreteGpu,
        );

        assert_eq!(
            format_directml_device_label(&info),
            "DirectML (NVIDIA GeForce RTX 4070 SUPER)"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn falls_back_to_vendor_name_when_adapter_name_is_missing() {
        let info = adapter_info("", NVIDIA_VENDOR_ID, wgpu::DeviceType::DiscreteGpu);

        assert_eq!(format_directml_device_label(&info), "DirectML (NVIDIA)");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn uses_adapter_name_for_cuda_label() {
        let info = adapter_info(
            "NVIDIA GeForce RTX 4070 SUPER",
            NVIDIA_VENDOR_ID,
            wgpu::DeviceType::DiscreteGpu,
        );

        assert_eq!(
            format_cuda_device_label(&info),
            "CUDA (NVIDIA GeForce RTX 4070 SUPER)"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn appends_runtime_note_to_device_label() {
        assert_eq!(
            annotate_device_label(
                "DirectML (NVIDIA GeForce RTX 2050)".into(),
                "CUDA unavailable"
            ),
            "DirectML (NVIDIA GeForce RTX 2050) [CUDA unavailable]"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn cuda_vendor_preference_only_targets_nvidia_or_unknown() {
        assert!(should_prefer_cuda_for_vendor(Some(NVIDIA_VENDOR_ID)));
        assert!(should_prefer_cuda_for_vendor(None));
        assert!(!should_prefer_cuda_for_vendor(Some(AMD_VENDOR_ID)));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn prefers_hardware_gpu_over_microsoft_software_adapter() {
        let discrete_gpu = adapter_info(
            "NVIDIA GeForce RTX 4070 SUPER",
            NVIDIA_VENDOR_ID,
            wgpu::DeviceType::DiscreteGpu,
        );
        let software_adapter = adapter_info(
            "Microsoft Basic Render Driver",
            MICROSOFT_VENDOR_ID,
            wgpu::DeviceType::VirtualGpu,
        );

        assert!(
            score_adapter_for_directml(&discrete_gpu)
                > score_adapter_for_directml(&software_adapter)
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn prefers_nvidia_over_equivalent_amd_adapter() {
        let nvidia_gpu = adapter_info(
            "NVIDIA GeForce RTX 4070 SUPER",
            NVIDIA_VENDOR_ID,
            wgpu::DeviceType::DiscreteGpu,
        );
        let amd_gpu = adapter_info(
            "Radeon RX 7800 XT",
            AMD_VENDOR_ID,
            wgpu::DeviceType::DiscreteGpu,
        );

        assert!(score_adapter_for_directml(&nvidia_gpu) > score_adapter_for_directml(&amd_gpu));
    }
}
