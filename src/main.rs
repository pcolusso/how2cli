use anyhow::{Context, Result};
use clap::Parser;
use llm::models::Llama;
use llm::{Model, Prompt};
use std::path::PathBuf;

const SYSTEM_PROMPT: &str = "[INST] <<SYS>>Provide only a bash command for the user's query. Do not post instructions on how to use the command. Output only the command that best matches the user's query. Be as succinct as possible.<</SYS>>";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(trailing_var_arg = true)]
    query: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let query = args.query.join(" ");

    let model_env = std::env::var("MODEL_PATH").context("MODEL_PATH env var not set.")?;
    let model_path = PathBuf::from(model_env);

    let parameters = llm::ModelParameters {
        prefer_mmap: false,
        context_size: 2048,
        lora_adapters: None,
        use_gpu: true,
        gpu_layers: None,
        rope_overrides: None,
        n_gqa: None,
    };


    let llama = llm::load::<Llama>(
        &model_path,
        llm::TokenizerSource::Embedded,
        parameters,
        llm::load_progress_callback_stdout,
    )
    .context("Rustformers had issues loading the provided model")?;

    let mut session = llama.start_session(Default::default());

    let prompt_full = format!("{SYSTEM_PROMPT} {query} [/INST]");

    let prompt = Prompt::Text(&prompt_full);

    let mut generated_tokens = String::with_capacity(2048);

    let _res = session.infer::<std::convert::Infallible>(
        &llama,
        // randomness provider
        &mut rand::thread_rng(),
        // the prompt to use for text generation, as well as other
        // inference parameters
        &llm::InferenceRequest {
            prompt: prompt,
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: Some(1024),
        },
        // llm::OutputRequest
        &mut Default::default(),
        // output callback
        |r| match r {
            llm::InferenceResponse::InferredToken(t) => {
                generated_tokens.push_str(&t);
                print!("{}", t);
                Ok(llm::InferenceFeedback::Continue)
            },
            llm::InferenceResponse::EotToken => {
                Ok(llm::InferenceFeedback::Halt)
            },
            _ => {
                Ok(llm::InferenceFeedback::Continue)
            },
        },
    ).context("Inference failed")?;

    println!("{generated_tokens}");
    Ok(())
}
