use std::path::Path;

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use ndarray::{Array2, Array3, ArrayView1, ArrayView3, Axis, Ix3};
use ort::ep::ExecutionProvider;
use ort::ep::directml::PerformancePreference;
use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
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
const NVIDIA_VENDOR_ID: u32 = 0x10DE;
const AMD_VENDOR_ID: u32 = 0x1002;
const AMD_CPU_VENDOR_ID: u32 = 0x1022;
const INTEL_VENDOR_ID: u32 = 0x8086;
const MICROSOFT_VENDOR_ID: u32 = 0x1414;
const QUALCOMM_VENDOR_ID: u32 = 0x5143;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetLang {
    En,
    Tr,
}

struct ModelBundle {
    encoder: Session,
    decoder: Session,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    pad_token_id: i64,
    eos_token_id: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecoderConfig {
    max_input_tokens: usize,
    max_new_tokens: usize,
    num_beams: usize,
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

pub struct Translator {
    tr_en: ModelBundle,
    en_tr: ModelBundle,
    decoder_config: DecoderConfig,
    device_label: String,
}

impl Translator {
    pub fn new() -> Result<Self> {
        let device_label = configure_ort()?;
        let decoder_config = DecoderConfig::from_env();

        let api = Api::new()?;
        let tr_en = load_model_pair(&api, TR_EN_ONNX_REPO_ID, TR_EN_BASE_REPO_ID)?;
        let en_tr = load_model_pair(&api, EN_TR_ONNX_REPO_ID, EN_TR_BASE_REPO_ID)?;

        Ok(Self {
            tr_en,
            en_tr,
            decoder_config,
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

        let model = match target {
            TargetLang::En => &mut self.tr_en,
            TargetLang::Tr => &mut self.en_tr,
        };

        let translated: Vec<String> = layout
            .translatable_segments
            .iter()
            .map(|segment| translate_segment(model, self.decoder_config, segment))
            .collect::<Result<_>>()?;

        Ok(layout.rebuild(translated))
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }
}

fn configure_ort() -> Result<String> {
    let force_cpu = std::env::var("TRANSLATE_DEVICE")
        .map(|value| value.trim().eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);

    if force_cpu {
        ort::init()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit();
        return Ok("CPU".to_owned());
    }

    let directml = DirectMLExecutionProvider::default()
        .with_performance_preference(PerformancePreference::HighPerformance);
    let use_directml = directml.is_available().unwrap_or(false);

    ort::init()
        .with_execution_providers([directml.build(), CPUExecutionProvider::default().build()])
        .commit();

    if use_directml {
        Ok(detect_directml_device_label())
    } else {
        Ok("CPU".to_owned())
    }
}

fn detect_directml_device_label() -> String {
    directml_adapter_info()
        .map(|info| format_directml_device_label(&info))
        .unwrap_or_else(|| "DirectML (GPU)".to_owned())
}

fn directml_adapter_info() -> Option<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..Default::default()
    });

    instance
        .enumerate_adapters(wgpu::Backends::DX12)
        .into_iter()
        .map(|adapter| adapter.get_info())
        .filter(|info| info.device_type != wgpu::DeviceType::Cpu)
        .max_by_key(score_adapter_for_directml)
}

fn score_adapter_for_directml(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let device_priority = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 5,
        wgpu::DeviceType::IntegratedGpu => 4,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 1,
    };
    let preferred_vendor = u8::from(!is_software_or_microsoft_adapter(info));

    (device_priority, preferred_vendor)
}

fn is_software_or_microsoft_adapter(info: &wgpu::AdapterInfo) -> bool {
    info.vendor == MICROSOFT_VENDOR_ID || info.name.contains("Microsoft")
}

fn format_directml_device_label(info: &wgpu::AdapterInfo) -> String {
    let adapter_name = info.name.trim();
    if !adapter_name.is_empty() {
        return format!("DirectML ({adapter_name})");
    }

    let vendor_name = gpu_vendor_name(info.vendor);
    format!("DirectML ({vendor_name})")
}

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

fn parse_env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn load_model_pair(api: &Api, onnx_repo_id: &str, base_repo_id: &str) -> Result<ModelBundle> {
    let onnx_model = api.model(onnx_repo_id.to_owned());
    let encoder_path = onnx_model.get("onnx/encoder_model.onnx")?;
    let decoder_path = onnx_model.get("onnx/decoder_model.onnx")?;
    let tokenizer_json_path = onnx_model.get("tokenizer.json")?;

    let base_model = api.model(base_repo_id.to_owned());
    let source_spm_path = base_model.get("source.spm")?;
    let target_spm_path = base_model.get("target.spm")?;
    let vocab_path = base_model.get("vocab.json")?;

    let encoder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_parallel_execution(true)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(&encoder_path)?;
    let decoder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_parallel_execution(true)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(&decoder_path)?;
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
        source_tokenizer,
        target_tokenizer,
        pad_token_id,
        eos_token_id,
    })
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
    let ids_t = TensorRef::from_array_view(ids.view()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mask_t = TensorRef::from_array_view(mask.view()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let out = encoder.run(ort::inputs![
        "input_ids" => ids_t,
        "attention_mask" => mask_t
    ])?;

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

    let out = decoder.run(ort::inputs![
        "input_ids" => target_t,
        "encoder_hidden_states" => hidden_t,
        "encoder_attention_mask" => enc_mask_t
    ])?;
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
    use super::{
        DecoderConfig, MICROSOFT_VENDOR_ID, NVIDIA_VENDOR_ID, TranslationLayout,
        format_directml_device_label, parse_env_usize, score_adapter_for_directml,
    };

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
        assert_eq!(DecoderConfig::default().num_beams, 1);
    }

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

    #[test]
    fn falls_back_to_vendor_name_when_adapter_name_is_missing() {
        let info = adapter_info("", NVIDIA_VENDOR_ID, wgpu::DeviceType::DiscreteGpu);

        assert_eq!(format_directml_device_label(&info), "DirectML (NVIDIA)");
    }

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
}
